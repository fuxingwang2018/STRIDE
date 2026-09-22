"""
Trainer for STRIDE.

This module provides the first clean training loop for STRIDE. It keeps the
useful structure from the legacy training setup - config-driven orchestration,
EMA support, checkpointing, validation, and resume support - while dropping the
legacy clutter tied to old dataset contracts, old generative paradigms,
classifier-free guidance, legacy project-specific branches, and other
branches that do not belong in STRIDE v1.

Responsibilities
----------------
- read the training YAML
- build model, loss, data loaders, optimizer, scheduler, and EMA
- move model / batches to device
- run training and validation epochs
- checkpoint periodically
- optionally resume from checkpoint
- log training progress summaries

Non-responsibilities
--------------------
- dataset internals
- transform definitions
- model architecture definitions
- sampler implementation
- full evaluation pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from stride_core.configs.model_config import ModelSpec
from stride_core.models.build_model import build_model
from stride_core.models.edm_loss import EDMLoss
from stride_core.training.checkpoints import (
    CheckpointConfig,
    LoadedCheckpointState,
    ensure_checkpoint_dir,
    load_checkpoint,
    resolve_resume_checkpoint,
    save_checkpoint,
)
from stride_core.training.data import BuiltTrainingData, build_training_data
from stride_core.training.ema import EMA, EMAConfig
from stride_core.training.optim import (
    OptimizerConfig,
    SchedulerConfig,
    build_optimizer,
    build_scheduler,
)

from stride_core.training.training_logging import (
    TrainingHistory,
    format_epoch_summary,
    format_step_progress,
    log_epoch_summary,
    log_run_setup,
    log_step_progress,
    print_section,
    summarize_run_setup,
)

from stride_core.generation.training_preview import (
    TrainingPreviewConfig,
    run_training_preview,
    should_run_training_preview,
)

from stride_core.training.monitoring import (
    MonitoringConfig,
    compute_preview_metrics,
    plot_default_history_metrics,
    should_run_monitoring,
)


# -----------------------------------------------------------------------------
# Training config
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRunConfig:
    """
    Minimal top-level trainer configuration extracted from the training YAML.
    """

    training_config_path: Path
    run_name: str
    output_dir: Path
    seed: int

    model_config_path: Path
    dataset_config_path: Path
    generation_config_path: Path | None

    accelerator: str

    max_epochs: int
    max_train_batches: int | None
    max_val_batches: int | None
    validate_every_n_epochs: int

    print_every_n_steps: int | None
    log_parameter_count: bool
    log_batch_shapes_once: bool
    print_step_progress: bool
    save_history_json: bool

    @classmethod
    def from_yaml(cls, training_config_path: str | Path) -> "TrainingRunConfig":
        path = Path(training_config_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Training config does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(
                f"Expected training config YAML to load into a dict, got {type(cfg)}"
            )

        return cls.from_dict(cfg, training_config_path=path)

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        training_config_path: str | Path,
    ) -> "TrainingRunConfig":
        training_cfg = _get_training_section(cfg)

        run_cfg = training_cfg.get("run", {})
        configs_cfg = training_cfg.get("configs", {})
        device_cfg = training_cfg.get("device", {})
        loop_cfg = training_cfg.get("loop", {})
        logging_cfg = training_cfg.get("logging", {})

        if not isinstance(run_cfg, dict):
            raise ValueError("Expected 'training.run' to be a dict")
        if not isinstance(configs_cfg, dict):
            raise ValueError("Expected 'training.configs' to be a dict")
        if not isinstance(device_cfg, dict):
            raise ValueError("Expected 'training.device' to be a dict")
        if not isinstance(loop_cfg, dict):
            raise ValueError("Expected 'training.loop' to be a dict")
        if not isinstance(logging_cfg, dict):
            raise ValueError("Expected 'training.logging' to be a dict")

        training_config_path = Path(training_config_path).resolve()
        repo_root = training_config_path.parents[2]

        run_name = str(run_cfg.get("name", "stride_training"))
        output_dir_raw = run_cfg.get("output_dir", "runs/stride_training")
        output_dir = _resolve_relative_to_repo(output_dir_raw, repo_root)
        seed = int(run_cfg.get("seed", 42))

        model_config_raw = configs_cfg.get("model_config")
        dataset_config_raw = configs_cfg.get("dataset_config")
        generation_config_raw = configs_cfg.get("generation_config", None)

        if model_config_raw is None:
            raise KeyError("Missing required key 'training.configs.model_config'")
        if dataset_config_raw is None:
            raise KeyError("Missing required key 'training.configs.dataset_config'")

        model_config_path = _resolve_relative_to_repo(model_config_raw, repo_root)
        dataset_config_path = _resolve_relative_to_repo(dataset_config_raw, repo_root)
        generation_config_path = (
            None
            if generation_config_raw is None
            else _resolve_relative_to_repo(generation_config_raw, repo_root)
        )

        accelerator = str(device_cfg.get("accelerator", "auto")).lower()
        if accelerator not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                f"training.device.accelerator must be 'auto', 'cpu', or 'cuda', got {accelerator}"
            )

        max_epochs = int(loop_cfg.get("max_epochs", 1))
        validate_every_n_epochs = int(loop_cfg.get("validate_every_n_epochs", 1))
        max_train_batches = _maybe_int(loop_cfg.get("max_train_batches", None))
        max_val_batches = _maybe_int(loop_cfg.get("max_val_batches", None))

        print_every_n_steps = _maybe_int(logging_cfg.get("print_every_n_steps", None))
        log_parameter_count = bool(logging_cfg.get("log_parameter_count", True))
        log_batch_shapes_once = bool(logging_cfg.get("log_batch_shapes_once", True))
        print_step_progress_flag = bool(logging_cfg.get("print_step_progress", False))
        save_history_json = bool(logging_cfg.get("save_history_json", True))

        if max_epochs <= 0:
            raise ValueError(f"training.loop.max_epochs must be > 0, got {max_epochs}")
        if validate_every_n_epochs <= 0:
            raise ValueError(
                "training.loop.validate_every_n_epochs must be > 0, "
                f"got {validate_every_n_epochs}"
            )

        return cls(
            training_config_path=training_config_path,
            run_name=run_name,
            output_dir=output_dir,
            seed=seed,
            model_config_path=model_config_path,
            dataset_config_path=dataset_config_path,
            generation_config_path=generation_config_path,
            accelerator=accelerator,
            max_epochs=max_epochs,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            validate_every_n_epochs=validate_every_n_epochs,
            print_every_n_steps=print_every_n_steps,
            log_parameter_count=log_parameter_count,
            log_batch_shapes_once=log_batch_shapes_once,
            print_step_progress=print_step_progress_flag,
            save_history_json=save_history_json,
        )


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class Trainer:
    """
    First clean STRIDE trainer.
    """

    def __init__(self, training_config_path: str | Path) -> None:
        self.cfg = TrainingRunConfig.from_yaml(training_config_path)

        self.device = self._resolve_device(self.cfg.accelerator)
        # Disable AMP for now. On the current LUMI setup this has shown unstable
        # behaviour leading to non-finite losses during EDM training.
        self.use_amp = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.checkpoint_dir = ensure_checkpoint_dir(self.cfg.output_dir / "checkpoints")

        self._set_seed(self.cfg.seed)

        print('Trainer: self.cfg.training_config_path:', self.cfg.training_config_path)
        self.data: BuiltTrainingData = build_training_data(self.cfg.training_config_path)

        self.model_spec = ModelSpec.from_yaml(self.cfg.model_config_path)
        self.model = build_model(self.model_spec).to(self.device)
        self.loss_fn = EDMLoss(
            rain_gate_cfg={
                "enabled": self.model_spec.rain_gate_loss.enabled,
                "loss_weight": self.model_spec.rain_gate_loss.loss_weight,
                "wet_threshold_mm": self.model_spec.rain_gate_loss.wet_threshold_mm,
                "target_variable": self.model_spec.rain_gate_loss.target_variable,
                "use_loss_reweighting": self.model_spec.rain_gate_loss.use_loss_reweighting,
                "reweight_detach": self.model_spec.rain_gate_loss.reweight_detach,
                "reweight_power": self.model_spec.rain_gate_loss.reweight_power,
            }
        )

        self.optimizer_config = OptimizerConfig.from_training_yaml(
            self.cfg.training_config_path
        )
        self.scheduler_config = SchedulerConfig.from_training_yaml(
            self.cfg.training_config_path
        )
        self.ema_config = EMAConfig.from_training_yaml(self.cfg.training_config_path)
        self.checkpoint_config = CheckpointConfig.from_training_yaml(
            self.cfg.training_config_path
        )
        self.preview_config = TrainingPreviewConfig.from_training_yaml(
            self.cfg.training_config_path
        )
        with open(self.cfg.training_config_path, "r", encoding="utf-8") as f:
            training_cfg_raw = yaml.safe_load(f)
        if not isinstance(training_cfg_raw, dict):
            raise ValueError(
                "Expected training config YAML to decode into a dict when building monitoring config"
            )
        self.monitoring_config = MonitoringConfig.from_training_dict(training_cfg_raw)

        self.optimizer = build_optimizer(self.model, self.optimizer_config)
        self.scheduler = build_scheduler(self.optimizer, self.scheduler_config)

        self.ema: EMA | None = None
        if self.ema_config.enabled:
            self.ema = EMA.from_config(self.model, self.ema_config)

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss: float | None = None

        self.history = TrainingHistory()
        self.history_path = self.cfg.output_dir / "history.json"
        self._did_log_batch_shapes = False
        self.monitoring_dir = self.cfg.output_dir / self.monitoring_config.output_subdir

        if self.checkpoint_config.resume_from is not None:
            self._resume_if_requested()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """
        Run the full training loop.
        """
        self._print_run_summary()

        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.cfg.max_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(epoch)
            self._print_epoch_summary("train", epoch, train_metrics)
            self.history.add_train_epoch(
                epoch + 1,
                {
                    **train_metrics,
                    "lr": self._get_current_lr(),
                },
            )

            val_metrics: dict[str, float] | None = None
            improved_val = False
            do_validate = (epoch + 1) % self.cfg.validate_every_n_epochs == 0
            if do_validate:
                val_metrics = self.validate_epoch(epoch)
                improved_val = self._update_best_val_loss(val_metrics["loss"])
                self._print_epoch_summary(
                    "valid",
                    epoch,
                    val_metrics,
                    improved=improved_val,
                )
                self.history.add_valid_epoch(
                    epoch + 1,
                    {
                        **val_metrics,
                        "improved": improved_val,
                        "best_val_loss": self.best_val_loss,
                    },
                )
                self._step_scheduler_after_validation(val_metrics["loss"])
            else:
                self._step_scheduler_without_metric()

            self._checkpoint_if_needed(
                epoch,
                train_metrics,
                val_metrics,
                improved_val=improved_val,
            )

            preview_result: dict[str, Any] | None = None
            if should_run_training_preview(
                epoch=epoch,
                improved_val=improved_val,
                preview_config=self.preview_config,
            ):
                preview_result = self._run_generation_preview(epoch)

            if (
                preview_result is not None
                and val_metrics is not None
                and should_run_monitoring(
                    epoch=epoch,
                    monitoring_config=self.monitoring_config,
                )
            ):
                preview_metrics = self._compute_preview_metrics(preview_result)
                self._attach_preview_metrics_to_last_valid_history(preview_metrics)

            if self.cfg.save_history_json:
                self.history.save_json(self.history_path)

            if self.monitoring_config.enabled and self.monitoring_config.save_history_plots:
                self._save_monitoring_history_plots()


    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()

        total_loss = 0.0
        #total_loss_tensor = 0.0
        total_loss = torch.zeros(1, device=self.device)
        num_batches = 0

        for batch_idx, batch in enumerate(self.data.train_loader):
            if (
                self.cfg.max_train_batches is not None
                and batch_idx >= self.cfg.max_train_batches
            ):
                break

            batch = self._move_batch_to_device(batch, self.device)

            if self.cfg.log_batch_shapes_once and not self._did_log_batch_shapes:
                self._print_batch_overview(batch)
                self._did_log_batch_shapes = True

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self.loss_fn(model=self.model, batch=batch)
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(
                        f"Expected loss_fn to return a torch.Tensor, got {type(loss)}"
                    )
                if loss.ndim != 0:
                    raise ValueError(
                        f"Expected scalar loss, got shape {tuple(loss.shape)}"
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Encountered non-finite training loss "
                        f"at epoch={epoch + 1}, batch_idx={batch_idx}, global_step={self.global_step}"
                    )

            self.scaler.scale(loss).backward()
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            loss_value = float(loss.item())
            total_loss += loss.detach()  # GPU accumulation, no sync
            #total_loss += loss_value
            num_batches += 1
            self.global_step += 1
            #with torch.no_grad():
            #    total_loss_tensor += loss.detach()

            # At end of epoch:
            #avg_loss = float(total_loss_tensor.item()) / num_batches


            should_print_step = (
                self.cfg.print_step_progress
                and self.cfg.print_every_n_steps is not None
                and self.cfg.print_every_n_steps > 0
                and self.global_step % self.cfg.print_every_n_steps == 0
            )
            if should_print_step:
                self._print_step_progress(
                    stage="train",
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_batches_total=len(self.data.train_loader),
                    loss_value=loss_value,
                    global_step=self.global_step,
                )

        if num_batches == 0:
            raise RuntimeError("No training batches were processed")

        return {
            #"loss": total_loss / num_batches,
            #"loss": avg_loss,
            "loss": (total_loss / num_batches).item(),  # single GPU→CPU sync per epoch
            "num_batches": float(num_batches),
        }

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> dict[str, float]:
        eval_model = self.model
        if self.ema is not None:
            eval_model = self.ema.clone_model(self.model).to(self.device)
        eval_model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.data.val_loader):
            if (
                self.cfg.max_val_batches is not None
                and batch_idx >= self.cfg.max_val_batches
            ):
                break

            batch = self._move_batch_to_device(batch, self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self.loss_fn(model=eval_model, batch=batch)
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(
                        f"Expected loss_fn to return a torch.Tensor, got {type(loss)}"
                    )
                if loss.ndim != 0:
                    raise ValueError(
                        f"Expected scalar loss, got shape {tuple(loss.shape)}"
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Encountered non-finite validation loss "
                        f"at epoch={epoch + 1}, batch_idx={batch_idx}"
                    )

            loss_value = float(loss.item())
            total_loss += loss_value
            num_batches += 1

            should_print_step = (
                self.cfg.print_step_progress
                and self.cfg.print_every_n_steps is not None
                and self.cfg.print_every_n_steps > 0
            )
            if should_print_step:
                self._print_step_progress(
                    stage="valid",
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_batches_total=len(self.data.val_loader),
                    loss_value=loss_value,
                    global_step=self.global_step,
                )

        if num_batches == 0:
            raise RuntimeError("No validation batches were processed")

        return {
            "loss": total_loss / num_batches,
            "num_batches": float(num_batches),
        }

    # ------------------------------------------------------------------
    # resume / checkpoint
    # ------------------------------------------------------------------

    def _resume_if_requested(self) -> None:
        resume_path = resolve_resume_checkpoint(
            self.checkpoint_config.resume_from,
            checkpoint_dir=self.checkpoint_dir,
        )
        if resume_path is None:
            return

        loaded: LoadedCheckpointState = load_checkpoint(
            resume_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            map_location="cpu",
        )

        self.current_epoch = loaded.epoch
        self.global_step = loaded.global_step
        self.best_val_loss = loaded.best_val_loss

        log_run_setup(
            [
                f"Resumed checkpoint: {loaded.checkpoint_path}",
                f"Resumed epoch:      {loaded.epoch}",
                f"Resumed global_step:{loaded.global_step}",
                f"Resumed best_val:   {loaded.best_val_loss}",
            ]
        )

    def _checkpoint_if_needed(
        self,
        epoch: int,
        train_metrics: dict[str, float] | None,
        val_metrics: dict[str, float] | None,
        *,
        improved_val: bool,
    ) -> None:
        if not self.checkpoint_config.enabled:
            return

        checkpoint_kwargs = {
            "model": self.model,
            "optimizer": self.optimizer,
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "scheduler": self.scheduler,
            "ema": self.ema,
            "best_val_loss": self.best_val_loss,
            "training_config_path": self.cfg.training_config_path,
            "model_config_path": self.cfg.model_config_path,
            "dataset_config_path": self.cfg.dataset_config_path,
            "generation_config_path": self.cfg.generation_config_path,
            "extra_state": {
                "run_name": self.cfg.run_name,
                "train_loss": None if train_metrics is None else train_metrics.get("loss"),
                "val_loss": None if val_metrics is None else val_metrics.get("loss"),
            },
        }

        # Always keep a latest checkpoint for resume safety.
        latest_paths = save_checkpoint(
            self.checkpoint_dir,
            **checkpoint_kwargs,
            tag=None,
            save_latest=False,
        )
        for label, path in latest_paths.items():
            log_run_setup([f"Saved checkpoint [latest/{label}]: {path}"])

        # Save periodic epoch snapshots only when the configured interval is reached.
        should_save_periodic = (
            self.checkpoint_config.save_every_n_epochs > 0
            and (epoch + 1) % self.checkpoint_config.save_every_n_epochs == 0
        )
        if should_save_periodic:
            periodic_paths = save_checkpoint(
                self.checkpoint_dir,
                **checkpoint_kwargs,
                tag=f"epoch_{epoch + 1:04d}",
                save_latest=False,
            )
            for label, path in periodic_paths.items():
                log_run_setup([f"Saved checkpoint [periodic/{label}]: {path}"])

        # Save a dedicated best checkpoint only when validation improved.
        if self.checkpoint_config.save_best and improved_val:
            best_paths = save_checkpoint(
                self.checkpoint_dir,
                **checkpoint_kwargs,
                tag="best",
                save_latest=False,
            )
            for label, path in best_paths.items():
                log_run_setup([f"Saved checkpoint [best/{label}]: {path}"])

    # ------------------------------------------------------------------
    # scheduler handling
    # ------------------------------------------------------------------

    def _step_scheduler_after_validation(self, val_loss: float) -> None:
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _step_scheduler_without_metric(self) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return
        self.scheduler.step()

    # ------------------------------------------------------------------
    # bookkeeping helpers
    # ------------------------------------------------------------------

    def _update_best_val_loss(self, val_loss: float) -> bool:
        improved = self.best_val_loss is None or val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = float(val_loss)
        return improved

    def _print_run_summary(self) -> None:
        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )

        print_section("STRIDE trainer start")

        total_params_to_show = total_params if self.cfg.log_parameter_count else 0
        trainable_params_to_show = (
            trainable_params if self.cfg.log_parameter_count else 0
        )

        setup_lines = summarize_run_setup(
            run_name=self.cfg.run_name,
            device=str(self.device),
            output_dir=self.cfg.output_dir,
            checkpoint_dir=self.checkpoint_dir,
            model_config_path=self.cfg.model_config_path,
            dataset_config_path=self.cfg.dataset_config_path,
            generation_config_path=self.cfg.generation_config_path,
            train_batches=len(self.data.train_loader),
            valid_batches=len(self.data.val_loader),
            total_params=total_params_to_show,
            trainable_params=trainable_params_to_show,
            ema_enabled=self.ema is not None,
        )
        setup_lines.extend(
            [
                f"Generation preview: {self.preview_config.enabled}",
                f"Monitoring:         {self.monitoring_config.enabled}",
                f"RainGate model:     {self.model_spec.rain_gate_model.enabled}",
                f"RainGate loss:      {self.model_spec.rain_gate_loss.enabled}",
            ]
        )
        if self.model_spec.rain_gate_loss.enabled:
            setup_lines.extend(
                [
                    f"RainGate weight:    {self.model_spec.rain_gate_loss.loss_weight}",
                    "RainGate wet thr.: "
                    f"{self.model_spec.rain_gate_loss.wet_threshold_mm} mm",
                ]
            )
        log_run_setup(setup_lines)

    def _print_epoch_summary(
        self,
        stage: str,
        epoch: int,
        metrics: dict[str, float],
        *,
        improved: bool | None = None,
    ) -> None:
        current_lr = self._get_current_lr()
        best_val_loss: float | None = None

        if stage == "valid":
            best_val_loss = self.best_val_loss

        log_epoch_summary(
            format_epoch_summary(
                stage,
                epoch,
                metrics,
                lr=current_lr,
                best_val_loss=best_val_loss,
                improved=improved,
            )
        )

    def _print_step_progress(
        self,
        *,
        stage: str,
        epoch: int,
        batch_idx: int,
        num_batches_total: int,
        loss_value: float,
        global_step: int | None = None,
    ) -> None:
        log_step_progress(
            format_step_progress(
                stage,
                epoch=epoch,
                batch_idx=batch_idx,
                num_batches_total=num_batches_total,
                loss_value=loss_value,
                global_step=global_step,
            )
        )

    def _get_current_lr(self) -> float | None:
        if not hasattr(self.optimizer, "param_groups"):
            return None
        if len(self.optimizer.param_groups) == 0:
            return None
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def _print_batch_overview(self, batch: dict[str, Any]) -> None:
        print_section("First training batch overview")
        for key in ("target", "cond_dynamic", "cond_static", "time_features"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                log_run_setup(
                    [
                        f"{key}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
                    ]
                )
            else:
                log_run_setup([f"{key}: type={type(value).__name__}"])


    def _get_preview_model(self) -> torch.nn.Module:
        if self.preview_config.use_ema_model and self.ema is not None:
            return self.ema.clone_model(self.model).to(self.device)
        return self.model

    def _run_generation_preview(self, epoch: int) -> dict[str, Any] | None:
        if self.cfg.generation_config_path is None:
            log_run_setup(
                [
                    "Skipping generation preview because no generation_config_path is set."
                ]
            )
            return None

        preview_model = self._get_preview_model()
        preview_model.eval()

        preview_result = run_training_preview(
            model=preview_model,
            val_dataset=self.data.val_dataset,
            generation_config_path=self.cfg.generation_config_path,
            output_dir=self.cfg.output_dir,
            epoch=epoch,
            device=self.device,
            preview_config=self.preview_config,
        )

        arrays_path = preview_result.get("arrays_path")
        figure_path = preview_result.get("figure_path")

        preview_lines = [f"Saved generation preview for epoch {epoch + 1:04d}."]
        if arrays_path is not None:
            preview_lines.append(f"  Preview arrays: {arrays_path}")
        if figure_path is not None:
            preview_lines.append(f"  Preview figure: {figure_path}")
        log_run_setup(preview_lines)

        return preview_result


    def _compute_preview_metrics(
        self,
        preview_result: dict[str, Any],
    ) -> dict[str, Any]:
        preview_payload = preview_result.get("preview_payload")
        if not isinstance(preview_payload, dict):
            raise ValueError(
                "preview_result['preview_payload'] must be a dict to compute preview metrics"
            )

        metrics = compute_preview_metrics(
            preview_payload=preview_payload,
            monitoring_config=self.monitoring_config,
        )
        metric_lines = ["Preview monitoring metrics:"]
        metric_lines.extend([f"  {key}: {value}" for key, value in metrics.items()])
        log_run_setup(metric_lines)
        return metrics

    def _attach_preview_metrics_to_last_valid_history(
        self,
        preview_metrics: dict[str, Any],
    ) -> None:
        if len(self.history.valid_epochs) == 0:
            return
        self.history.valid_epochs[-1].update(preview_metrics)

    def _save_monitoring_history_plots(self) -> None:
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = plot_default_history_metrics(
            self.history_path,
            output_dir=self.monitoring_dir,
            smoothing_alpha=self.monitoring_config.smoothing_alpha,
            preview_metric_keys=self.monitoring_config.preview_metric_keys,
            preview_metrics_layout=self.monitoring_config.preview_metrics_layout,
        )
        for path in saved_paths:
            log_run_setup([f"Saved monitoring plot: {path}"])

    # ------------------------------------------------------------------
    # device / seed / batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(accelerator: str) -> torch.device:
        accelerator = accelerator.lower()
        if accelerator == "cpu":
            return torch.device("cpu")
        if accelerator == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            return torch.device("cuda")
        if accelerator == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raise ValueError(f"Unsupported accelerator value: {accelerator}")

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        def _move(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device, non_blocking=True)
            if isinstance(value, dict):
                return {k: _move(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_move(v) for v in value]
            return value

        return {key: _move(value) for key, value in batch.items()}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------



def _get_training_section(cfg: dict[str, Any]) -> dict[str, Any]:
    if "training" not in cfg:
        raise KeyError("Expected top-level key 'training' in training config")

    training_cfg = cfg["training"]
    if not isinstance(training_cfg, dict):
        raise ValueError(
            f"Expected 'training' section to be a dict, got {type(training_cfg)}"
        )

    return training_cfg



def _resolve_relative_to_repo(path_like: str | Path, repo_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()



def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
