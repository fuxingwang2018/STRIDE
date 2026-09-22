"""
Core generation orchestration for STRIDE.

This module is the generation-side analogue of `trainer.py`: it decides what to
load, which split to run on, which checkpoint weights to use, how many samples
to generate per case, and where outputs should be written.

Current scope
-------------
- load generation-run config from YAML
- infer checkpoint paths from a training config or use an explicit checkpoint
- resolve all referenced paths relative to the single generation-run config
- rebuild the model and load checkpoint weights
- build the requested dataset split and dataloader
- generate one or more samples per case
- save generated outputs as compressed `.npz`
- optionally save simple per-case metadata

Intentional non-goals for this first version
-------------------------------------------
- sophisticated distributed generation
- PMM computation implementation details
- rich plotting during full generation runs
- adapter-agnostic plugin discovery

The first milestone for this file is a clean, trustworthy path from:
    best checkpoint -> test split -> generated outputs on disk
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import logging
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np

logger = logging.getLogger(__name__)

from stride_core.configs.adapter_config import AdapterConfig
from stride_core.generation.generation_utils import (
    build_preview_payload,
    generate_batch,
    move_batch_to_device,
)
from stride_core.generation.output_saving import (
    case_output_dir as build_case_output_dir,
    save_aggregate,
    save_member,
    save_member_bundle,
    save_run_metadata,
)
from stride_core.generation.pmm import pmm_from_ensemble
from stride_core.models.build_model import build_model
from stride_core.training.data import build_dataset_adapter, stride_collate_fn


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRunPaths:
    dataset_config_path: Path
    model_config_path: Path
    generation_config_path: Path
    training_config_path: Path | None


@dataclass(frozen=True)
class CheckpointSelectionConfig:
    mode: str
    path: Path | None
    use_ema_weights: bool


@dataclass(frozen=True)
class GenerationDataConfig:
    split: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool


@dataclass(frozen=True)
class GenerationSamplingConfig:
    ensemble_size: int
    use_fixed_seed: bool
    base_seed: int


@dataclass(frozen=True)
class GenerationLimitsConfig:
    max_cases: int | None


@dataclass(frozen=True)
class GenerationOutputConfig:
    save_members: bool
    save_metadata: bool
    save_physical: bool
    save_ensemble_mean: bool
    save_pmm: bool
    save_plots: bool
    output_format: str
    storage_mode: str


@dataclass(frozen=True)
class GenerationRunConfig:
    run_name: str
    output_dir: Path
    device_accelerator: str
    paths: GenerationRunPaths
    checkpoint: CheckpointSelectionConfig
    data: GenerationDataConfig
    sampling: GenerationSamplingConfig
    limits: GenerationLimitsConfig
    outputs: GenerationOutputConfig
    config_path: Path

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "GenerationRunConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Generation run config does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(
                f"Expected generation run YAML to decode into a dict, got {type(cfg)}"
            )

        return cls.from_dict(cfg, config_path=path)

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        config_path: str | Path,
    ) -> "GenerationRunConfig":
        run_cfg = cfg.get("generation_run", cfg)
        if not isinstance(run_cfg, dict):
            raise ValueError("Expected generation run config to be a dict")

        paths_cfg = run_cfg.get("paths", run_cfg.get("configs", {}))
        if not isinstance(paths_cfg, dict):
            raise ValueError(
                "Expected 'generation_run.paths' or 'generation_run.configs' to be a dict"
            )

        dataset_config_path = cls._resolve_path(
            paths_cfg["dataset_config"],
            config_path=config_path,
        )
        model_config_path = cls._resolve_path(
            paths_cfg["model_config"],
            config_path=config_path,
        )
        generation_config_path = cls._resolve_path(
            paths_cfg["generation_config"],
            config_path=config_path,
        )
        training_config_raw = paths_cfg.get("training_config", None)
        training_config_path = (
            None
            if training_config_raw is None
            else cls._resolve_path(training_config_raw, config_path=config_path)
        )

        with open(generation_config_path, "r", encoding="utf-8") as f:
            generation_base_cfg = yaml.safe_load(f)
        if not isinstance(generation_base_cfg, dict):
            raise ValueError(
                f"Expected generation base config YAML to decode into a dict, got {type(generation_base_cfg)}"
            )

        base_generation_cfg = generation_base_cfg.get("generation", generation_base_cfg)
        if not isinstance(base_generation_cfg, dict):
            raise ValueError(
                "Expected top-level 'generation' section in generation base config to be a dict"
            )

        checkpoint_cfg = run_cfg.get("checkpoint", base_generation_cfg.get("checkpoint", {}))
        data_cfg = base_generation_cfg.get("data", {}).copy()
        data_cfg.update(run_cfg.get("data", {}))
        sampling_cfg = base_generation_cfg.get("sampling", {}).copy()
        sampling_cfg.update(run_cfg.get("sampling", {}))
        limits_cfg = base_generation_cfg.get("limits", {}).copy()
        limits_cfg.update(run_cfg.get("limits", {}))
        outputs_cfg = base_generation_cfg.get("outputs", {}).copy()
        outputs_cfg.update(run_cfg.get("outputs", {}))
        device_cfg = base_generation_cfg.get("device", {}).copy()
        device_cfg.update(run_cfg.get("device", {}))

        if not isinstance(checkpoint_cfg, dict):
            raise ValueError("Expected generation checkpoint config to be a dict")
        if not isinstance(data_cfg, dict):
            raise ValueError("Expected generation data config to be a dict")
        if not isinstance(sampling_cfg, dict):
            raise ValueError("Expected generation sampling config to be a dict")
        if not isinstance(limits_cfg, dict):
            raise ValueError("Expected generation limits config to be a dict")
        if not isinstance(outputs_cfg, dict):
            raise ValueError("Expected generation outputs config to be a dict")
        if not isinstance(device_cfg, dict):
            raise ValueError("Expected generation device config to be a dict")

        checkpoint_mode = str(checkpoint_cfg.get("mode", "best"))
        checkpoint_path_raw = checkpoint_cfg.get("path", paths_cfg.get("checkpoint_path", None))
        checkpoint_path = (
            None
            if checkpoint_path_raw is None
            else cls._resolve_path(checkpoint_path_raw, config_path=config_path)
        )
        if checkpoint_mode not in {"best", "latest", "explicit"}:
            raise ValueError(
                "generation_run.checkpoint.mode must be one of: "
                f"best, latest, explicit. Got {checkpoint_mode!r}."
            )
        if checkpoint_mode == "explicit" and checkpoint_path is None:
            raise ValueError(
                "generation_run.checkpoint.path must be set when mode='explicit'"
            )

        batch_size = int(data_cfg.get("batch_size", 1))
        if batch_size <= 0:
            raise ValueError(f"generation_run.data.batch_size must be > 0, got {batch_size}")

        ensemble_size = int(sampling_cfg.get("ensemble_size", 1))
        if ensemble_size <= 0:
            raise ValueError(
                f"generation_run.sampling.ensemble_size must be > 0, got {ensemble_size}"
            )

        max_cases_raw = limits_cfg.get("max_cases", None)
        max_cases = None if max_cases_raw is None else int(max_cases_raw)
        if max_cases is not None and max_cases <= 0:
            raise ValueError(
                f"generation_run.limits.max_cases must be > 0 when set, got {max_cases}"
            )

        output_format = str(outputs_cfg.get("output_format", "npz"))
        if output_format != "npz":
            raise ValueError(
                f"Only 'npz' output_format is currently supported, got {output_format!r}"
            )

        storage_mode = str(outputs_cfg.get("storage_mode", "per_member"))
        if storage_mode not in {"per_member", "per_member_bundle"}:
            raise ValueError(
                "generation_run.outputs.storage_mode must be one of: "
                f"'per_member', 'per_member_bundle'. Got {storage_mode!r}."
            )

        return cls(
            run_name=str(run_cfg.get("run_name", run_cfg.get("name", "generation_run"))),
            output_dir=cls._resolve_path(
                run_cfg.get(
                    "output_dir",
                    outputs_cfg.get("output_dir", "runs/generation_run"),
                ),
                config_path=config_path,
            ),
            device_accelerator=str(device_cfg.get("accelerator", "auto")),
            paths=GenerationRunPaths(
                dataset_config_path=dataset_config_path,
                model_config_path=model_config_path,
                generation_config_path=generation_config_path,
                training_config_path=training_config_path,
            ),
            checkpoint=CheckpointSelectionConfig(
                mode=checkpoint_mode,
                path=checkpoint_path,
                use_ema_weights=bool(checkpoint_cfg.get("use_ema_weights", True)),
            ),
            data=GenerationDataConfig(
                split=str(data_cfg.get("split", "test")),
                batch_size=batch_size,
                num_workers=int(data_cfg.get("num_workers", 0)),
                pin_memory=bool(data_cfg.get("pin_memory", False)),
                shuffle=bool(data_cfg.get("shuffle", False)),
            ),
            sampling=GenerationSamplingConfig(
                ensemble_size=ensemble_size,
                use_fixed_seed=bool(sampling_cfg.get("use_fixed_seed", False)),
                base_seed=int(sampling_cfg.get("base_seed", 42)),
            ),
            limits=GenerationLimitsConfig(
                max_cases=max_cases,
            ),
            outputs=GenerationOutputConfig(
                save_members=bool(outputs_cfg.get("save_members", True)),
                save_metadata=bool(outputs_cfg.get("save_metadata", True)),
                save_physical=bool(outputs_cfg.get("save_physical", True)),
                save_ensemble_mean=bool(outputs_cfg.get("save_ensemble_mean", False)),
                save_pmm=bool(outputs_cfg.get("save_pmm", False)),
                save_plots=bool(outputs_cfg.get("save_plots", False)),
                output_format=output_format,
                storage_mode=storage_mode,
            ),
            config_path=Path(config_path),
        )
    @staticmethod
    def _resolve_path(raw_path: str | Path, *, config_path: str | Path) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path.resolve()

        config_path = Path(config_path).resolve()
        repo_root = Path(__file__).resolve().parents[2]

        config_relative = (config_path.parent / path).resolve()
        if config_relative.exists():
            return config_relative

        repo_relative = (repo_root / path).resolve()
        if repo_relative.exists():
            return repo_relative

        if len(path.parts) > 0 and path.parts[0] in {
            "runs",
            "configs",
            "data_adapters",
            "stride_core",
        }:
            return repo_relative

        return config_relative

# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------


class Generator:
    """
    Config-driven generation pipeline.
    """

    def __init__(self, config: GenerationRunConfig | str | Path) -> None:
        self.cfg = (
            config if isinstance(config, GenerationRunConfig) else GenerationRunConfig.from_yaml(config)
        )

        self.output_dir = self.cfg.output_dir
        self.samples_dir = self.output_dir / "samples"
        self.metadata_path = self.output_dir / "generation_metadata.json"

        self.device = self._resolve_device(self.cfg.device_accelerator)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.adapter_cfg = AdapterConfig.from_yaml(self.cfg.paths.dataset_config_path)
        self.adapter = self._build_adapter()
        self.dataset = self._build_dataset()
        self.loader = self._build_dataloader()

        self.model = build_model(self.cfg.paths.model_config_path).to(self.device)
        self.model.eval()

        self.checkpoint_path = self._resolve_checkpoint_path()
        self.checkpoint_payload = self._load_checkpoint_into_model(self.checkpoint_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self) -> None:
        self._print_run_summary()
        self._save_run_metadata()

        generated_case_count = 0
        #print('run: self.loader=', self.loader)
        loader = self.loader
        ds = loader.dataset

        """
        print("=== DataLoader ===")
        print(f"  batch_size:         {loader.batch_size}")
        print(f"  num_workers:        {loader.num_workers}")
        print(f"  pin_memory:         {loader.pin_memory}")
        print(f"  drop_last:          {loader.drop_last}")
        print(f"  shuffle:            {isinstance(loader.sampler, torch.utils.data.RandomSampler)}")
        print(f"  persistent_workers: {loader.persistent_workers}")
        print(f"  prefetch_factor:    {loader.prefetch_factor if loader.num_workers > 0 else 'N/A'}")
        print(f"  collate_fn:         {loader.collate_fn}")

        print("\n=== Dataset ===")
        print(f"  type:               {type(ds).__name__}")
        print(f"  len:                {len(ds)}")
        print(f"dataset: len={len(ds)}, sample_keys={list(ds[0].keys())}, shapes={ {k: v.shape if hasattr(v, 'shape') else type(v).__name__ for k, v in ds[0].items()} }")
        """

        for batch_idx, batch in enumerate(self.loader):
            if (
                self.cfg.limits.max_cases is not None
                and generated_case_count >= self.cfg.limits.max_cases
            ):
                break
           
            #print('run: batch_idx=', batch_idx)
            batch = move_batch_to_device(batch, self.device)
            processed_cases = self._generate_for_batch(
                batch,
                batch_idx=batch_idx,
                remaining_cases=None
                if self.cfg.limits.max_cases is None
                else self.cfg.limits.max_cases - generated_case_count,
            )
            generated_case_count += processed_cases
            #print('run: processed_cases=', processed_cases)
            #print('run: generated_case_count=', generated_case_count)
            #print('run: self.cfg.limits.max_cases=', self.cfg.limits.max_cases)

        if self.cfg.outputs.save_plots:
            logger.info(
                "Full-run generation plotting was requested, but plotting is not implemented yet in this first generator version."
            )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, accelerator: str) -> torch.device:
        if accelerator == "cpu":
            return torch.device("cpu")
        if accelerator == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            return torch.device("cuda")
        if accelerator == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raise ValueError(
            f"Unsupported generation_run.device.accelerator={accelerator!r}. "
            "Expected one of: auto, cpu, cuda."
        )

    def _build_adapter(self) -> Any:
        adapter_cfg = replace(self.adapter_cfg, split_name=self.cfg.data.split)
        return build_dataset_adapter(adapter_cfg)

    def _build_dataset(self) -> Any:
        adapter = self.adapter
        build_dataset = getattr(adapter, "build_dataset", None)
        if not callable(build_dataset):
            raise AttributeError(
                "Dataset adapter must provide a callable build_dataset() method"
            )
        return build_dataset()

    def _build_dataloader(self) -> DataLoader[dict[str, Any]]:
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=self.cfg.data.shuffle,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=False,
            collate_fn=stride_collate_fn,
        )

    def _resolve_checkpoint_path(self) -> Path:
        if self.cfg.checkpoint.mode == "explicit":
            assert self.cfg.checkpoint.path is not None
            path = self.cfg.checkpoint.path
            if not path.exists():
                raise FileNotFoundError(f"Explicit checkpoint path does not exist: {path}")
            return path

        training_config_path = self.cfg.paths.training_config_path
        if training_config_path is None:
            raise ValueError(
                "generation_run.configs.training_config must be set when checkpoint.mode "
                "is not 'explicit'"
            )
        if not training_config_path.exists():
            raise FileNotFoundError(
                f"Training config for checkpoint inference does not exist: {training_config_path}"
            )

        with open(training_config_path, "r", encoding="utf-8") as f:
            training_cfg = yaml.safe_load(f)
        if not isinstance(training_cfg, dict):
            raise ValueError(
                f"Expected training config YAML to decode into a dict, got {type(training_cfg)}"
            )

        training_root = training_cfg.get("training", training_cfg)
        if not isinstance(training_root, dict):
            raise ValueError("Expected 'training' section in training config to be a dict")

        run_cfg = training_root.get("run", {})
        if not isinstance(run_cfg, dict):
            raise ValueError("Expected 'training.run' section to be a dict")

        training_output_dir_raw = run_cfg.get("output_dir", "runs/train_run")
        if not isinstance(training_output_dir_raw, str) or training_output_dir_raw.strip() == "":
            raise ValueError(
                "Expected 'training.run.output_dir' in training config to be a non-empty string"
            )
        training_output_dir = self.cfg._resolve_path(
            training_output_dir_raw,
            config_path=training_config_path,
        )
        checkpoint_dir = training_output_dir / "checkpoints"

        if self.cfg.checkpoint.mode == "best":
            path = checkpoint_dir / "checkpoint_best.pt"
        elif self.cfg.checkpoint.mode == "latest":
            path = checkpoint_dir / "checkpoint_latest.pt"
        else:
            raise ValueError(f"Unsupported checkpoint mode {self.cfg.checkpoint.mode!r}")

        if not path.exists():
            raise FileNotFoundError(
                f"Could not resolve requested checkpoint at: {path}"
            )
        return path

    def _looks_like_model_state_dict(self, payload: Any) -> bool:
        if not isinstance(payload, dict) or len(payload) == 0:
            return False
        return all(isinstance(key, str) for key in payload.keys()) and any(
            isinstance(value, torch.Tensor) for value in payload.values()
        )

    def _try_load_state_dict_candidate(self, candidate: Any) -> bool:
        if not self._looks_like_model_state_dict(candidate):
            return False
        self.model.load_state_dict(candidate)
        return True

    def _load_checkpoint_into_model(self, checkpoint_path: Path) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Expected checkpoint to load into a dict, got {type(checkpoint)}"
            )

        state_dict_loaded = False

        if self.cfg.checkpoint.use_ema_weights:
            ema_state_dict = checkpoint.get("ema_state_dict")
            state_dict_loaded = self._try_load_state_dict_candidate(ema_state_dict)

            if not state_dict_loaded:
                ema_payload = checkpoint.get("ema")
                if isinstance(ema_payload, dict):
                    shadow_params = ema_payload.get("shadow_params")
                    state_dict_loaded = self._try_load_state_dict_candidate(
                        shadow_params
                    )

        if not state_dict_loaded:
            model_state_dict = checkpoint.get("model_state_dict")
            state_dict_loaded = self._try_load_state_dict_candidate(model_state_dict)

        if not state_dict_loaded:
            model_payload = checkpoint.get("model")
            state_dict_loaded = self._try_load_state_dict_candidate(model_payload)

        if not state_dict_loaded:
            available_keys = sorted(str(key) for key in checkpoint.keys())
            raise KeyError(
                "Could not find a loadable model state in checkpoint. Tried keys: "
                "'ema_state_dict', 'ema.shadow_params', 'model_state_dict', 'model'. "
                f"Available top-level keys: {available_keys}"
            )

        self.model.eval()
        return checkpoint

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_for_batch(
        self,
        batch: dict[str, Any],
        *,
        batch_idx: int,
        remaining_cases: int | None,
    ) -> int:
        batch_size = self._infer_batch_size(batch)
        #print('_generate_for_batch: batch_size', batch_size)
        #print('_generate_for_batch: remaining_cases', remaining_cases)
        if remaining_cases is not None:
            batch_size = min(batch_size, remaining_cases)
        dates = self._get_batch_meta_list(batch, "date", batch_size)
        #print('_generate_for_batch: dates', dates)
        #print('_generate_for_batch: batch', batch)

        generated_members: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        for member_idx in range(self.cfg.sampling.ensemble_size):
            self._set_member_seed(batch_idx=batch_idx, member_idx=member_idx)
            generated = generate_batch(
                self.model,
                batch,
                generation_config=self.cfg.paths.generation_config_path,
                device=self.device,
            )
            if not isinstance(generated, torch.Tensor):
                raise TypeError(
                    f"generate_batch returned type {type(generated)}, expected torch.Tensor"
                )
            if generated.ndim != 4:
                raise ValueError(
                    f"Expected generated tensor to have shape (B, C, H, W), got {tuple(generated.shape)}"
                )

            for case_idx in range(batch_size):
                generated_members[case_idx].append(
                    generated[case_idx : case_idx + 1].detach().cpu()
                )

        #print('_generate_for_batch: batch_idx=', batch_idx)
        for case_idx in range(batch_size):
            case_batch = self._slice_batch(batch, case_idx)
            case_output_dir = build_case_output_dir(
                self.samples_dir,
                date=dates[case_idx],
                fallback_name=f"batch_{batch_idx:04d}_case_{case_idx:02d}",
            )

            if self.cfg.outputs.save_members:
                if self.cfg.outputs.storage_mode == "per_member":
                    for member_idx, generated_member in enumerate(generated_members[case_idx]):
                        save_member(
                            case_batch,
                            generated_member,
                            case_output_dir=case_output_dir,
                            member_idx=member_idx,
                            checkpoint_path=self.checkpoint_path,
                            save_physical=self.cfg.outputs.save_physical,
                            save_metadata=self.cfg.outputs.save_metadata,
                        )
                elif self.cfg.outputs.storage_mode == "per_member_bundle":
                    save_member_bundle(
                        case_batch,
                        generated_members[case_idx],
                        case_output_dir=case_output_dir,
                        checkpoint_path=self.checkpoint_path,
                        save_physical=self.cfg.outputs.save_physical,
                        save_metadata=self.cfg.outputs.save_metadata,
                    )
                else:
                    raise ValueError(
                        f"Unsupported storage_mode {self.cfg.outputs.storage_mode!r}"
                    )

            if self.cfg.outputs.save_ensemble_mean:
                ensemble_mean = self._compute_ensemble_mean(generated_members[case_idx])
                save_aggregate(
                    case_batch,
                    ensemble_mean,
                    case_output_dir=case_output_dir,
                    name="ensemble_mean",
                    checkpoint_path=self.checkpoint_path,
                    save_physical=self.cfg.outputs.save_physical,
                    save_metadata=self.cfg.outputs.save_metadata,
                )

            if self.cfg.outputs.save_pmm:
                pmm_field = self._compute_pmm(case_batch, generated_members[case_idx])
                save_aggregate(
                    case_batch,
                    pmm_field,
                    case_output_dir=case_output_dir,
                    name="pmm",
                    checkpoint_path=self.checkpoint_path,
                    save_physical=self.cfg.outputs.save_physical,
                    save_metadata=self.cfg.outputs.save_metadata,
                )

        return batch_size


    def _compute_ensemble_mean(self, generated_members: list[torch.Tensor]) -> torch.Tensor:
        if len(generated_members) == 0:
            raise ValueError("generated_members must contain at least one tensor")
        stacked = torch.stack(generated_members, dim=0)
        return torch.mean(stacked, dim=0)


    def _compute_pmm(
        self,
        batch: dict[str, Any],
        generated_members: list[torch.Tensor],
    ) -> torch.Tensor:
        if len(generated_members) == 0:
            raise ValueError("generated_members must contain at least one tensor")

        ensemble = torch.cat(generated_members, dim=0)  # [M, 1, H, W]
        if ensemble.ndim != 4 or ensemble.shape[1] != 1:
            raise ValueError(
                "PMM currently expects univariate generated members with shape [M, 1, H, W], "
                f"got {tuple(ensemble.shape)}"
            )

        ensemble_univariate = ensemble[:, 0]  # [M, H, W]
        mask = self._build_pmm_mask(batch, generated_members[0])
        return pmm_from_ensemble(
            ensemble_univariate,
            mask=mask,
            exclude_zeros=False,
        )

    def _build_pmm_mask(
        self,
        batch: dict[str, Any],
        reference_generated: torch.Tensor,
    ) -> torch.Tensor | None:
        payload = build_preview_payload(batch, reference_generated)
        cond_static_physical = payload.get("cond_static_physical")
        if not isinstance(cond_static_physical, torch.Tensor):
            return None

        static_variables = tuple(self.adapter_cfg.static_variables)
        if "lsm" not in static_variables:
            return None

        lsm_idx = static_variables.index("lsm")
        if cond_static_physical.ndim != 4:
            return None
        if lsm_idx >= cond_static_physical.shape[1]:
            return None

        lsm = cond_static_physical[:, lsm_idx]  # [B, H, W]
        return lsm > 0.5


    def _slice_batch(self, batch: dict[str, Any], case_idx: int) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value[case_idx : case_idx + 1]
            elif isinstance(value, dict):
                sliced_meta: dict[str, Any] = {}
                for meta_key, meta_value in value.items():
                    if isinstance(meta_value, list):
                        sliced_meta[meta_key] = [meta_value[case_idx]]
                    else:
                        sliced_meta[meta_key] = meta_value
                sliced[key] = sliced_meta
            else:
                sliced[key] = value
        return sliced


    # ------------------------------------------------------------------
    # Metadata / saving helpers
    # ------------------------------------------------------------------

    def _save_run_metadata(self) -> None:
        save_run_metadata(
            self.metadata_path,
            run_name=self.cfg.run_name,
            generation_run_config=self.cfg.config_path,
            dataset_config=self.cfg.paths.dataset_config_path,
            model_config=self.cfg.paths.model_config_path,
            generation_config=self.cfg.paths.generation_config_path,
            training_config=self.cfg.paths.training_config_path,
            checkpoint_mode=self.cfg.checkpoint.mode,
            checkpoint_path=self.checkpoint_path,
            use_ema_weights=self.cfg.checkpoint.use_ema_weights,
            split=self.cfg.data.split,
            batch_size=self.cfg.data.batch_size,
            ensemble_size=self.cfg.sampling.ensemble_size,
            max_cases=self.cfg.limits.max_cases,
            device=str(self.device),
            save_physical=self.cfg.outputs.save_physical,
            save_ensemble_mean=self.cfg.outputs.save_ensemble_mean,
            save_pmm=self.cfg.outputs.save_pmm,
            save_plots=self.cfg.outputs.save_plots,
            output_format=self.cfg.outputs.output_format,
            storage_mode=self.cfg.outputs.storage_mode,
        )


    def _infer_batch_size(self, batch: dict[str, Any]) -> int:
        target = batch.get("target")
        #print('_infer_batch_size:int(target.shape[0]),', int(target.shape[0]))
        if isinstance(target, torch.Tensor):
            return int(target.shape[0])
        cond_dynamic = batch.get("cond_dynamic")
        #print('_infer_batch_size:int(cond_dynamic.shape[0]),', int(cond_dynamic.shape[0]))
        if isinstance(cond_dynamic, torch.Tensor):
            return int(cond_dynamic.shape[0])
        raise ValueError("Could not infer batch size from batch tensors")

    def _get_batch_meta_list(
        self,
        batch: dict[str, Any],
        key: str,
        batch_size: int,
    ) -> list[Any]:
        meta = batch.get("meta", {})
        if not isinstance(meta, dict):
            return [None] * batch_size
        value = meta.get(key)
        if isinstance(value, list):
            if len(value) == batch_size:
                return value
            if len(value) == 1 and batch_size == 1:
                return value
        return [None] * batch_size

    def _set_member_seed(self, *, batch_idx: int, member_idx: int) -> None:
        if not self.cfg.sampling.use_fixed_seed:
            return
        seed = int(self.cfg.sampling.base_seed + 10_000 * batch_idx + member_idx)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32 - 1))

    def _safe_len(self, obj: Any) -> int | None:
        try:
            return len(obj)
        except TypeError:
            return None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _print_run_summary(self) -> None:
        logger.info("\n=======================")
        logger.info("STRIDE generation run")
        logger.info("=======================")
        logger.info(f"Run name:          {self.cfg.run_name}")
        logger.info(f"Device:            {self.device}")
        logger.info(f"Output dir:        {self.output_dir}")
        logger.info(f"Samples dir:       {self.samples_dir}")
        logger.info(f"Dataset config:    {self.cfg.paths.dataset_config_path}")
        logger.info(f"Model config:      {self.cfg.paths.model_config_path}")
        logger.info(f"Generation config: {self.cfg.paths.generation_config_path}")
        logger.info(f"Checkpoint path:   {self.checkpoint_path}")
        logger.info(f"Split:             {self.cfg.data.split}")
        logger.info(f"Shuffle:           {self.cfg.data.shuffle}")

        dataset_len = self._safe_len(self.dataset)
        loader_len = self._safe_len(self.loader)

        logger.info(
            f"Dataset length:    {'unknown' if dataset_len is None else dataset_len}"
        )
        logger.info(
            f"Num batches:       {'unknown' if loader_len is None else loader_len}"
        )
        logger.info(f"Ensemble size:     {self.cfg.sampling.ensemble_size}")
        logger.info(f"Max cases:         {self.cfg.limits.max_cases}")
        logger.info(f"Save members:      {self.cfg.outputs.save_members}")
        logger.info(f"Save ens. mean:    {self.cfg.outputs.save_ensemble_mean}")
        logger.info(f"Save PMM:          {self.cfg.outputs.save_pmm}")
        logger.info(f"Storage mode:      {self.cfg.outputs.storage_mode}")
        logger.info(f"Use EMA weights:   {self.cfg.checkpoint.use_ema_weights}")
