

"""
NorCP STRIDE adapter.

This adapter is intentionally kept close in spirit to the DANRA/ERA5 adapter so
it stays compatible with the rest of the current STRIDE setup.

Returned batch/sample contract
-----------------------------
{
    "target": torch.Tensor,         # [C_out, H, W]
    "cond_dynamic": torch.Tensor,   # [C_dyn, H_lr, W_lr]
    "cond_static": torch.Tensor | None,  # [C_static, H, W] or None
    "cond_coord": dict,
    "meta": dict,
    "time_features": torch.Tensor,  # [2] = [sin(DOY), cos(DOY)]
}

Design notes
------------
- The adapter is split-manifest driven.
- Sample indexing is delegated to `indexing.py`.
- Field loading is delegated to `features.py`.
- Statistics are loaded from saved JSON files and used to build per-variable
  transforms.
- Static variables are optional because some future NorCP scenarios do not have
  topography available.
- Fixed crop support is included in a lightweight form for later use, while the
  current intended default is full-domain statistics/training.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data_adapters.norcp.features import load_field_at_timestamp, load_static_field
from data_adapters.norcp.indexing import build_norcp_sample_index
from data_adapters.norcp.regions import CropSpec, derive_lr_crop_from_hr_crop, maybe_crop_3d
from data_adapters.norcp.splits.splits import get_split_timestamps, load_split_manifest
from data_adapters.norcp.statistics.io import build_stats_output_path, load_transform_stats
from data_adapters.norcp.transforms import BaseTransform, build_transform
from data_adapters.norcp.variable_registry import get_default_transform
from stride_core.configs.adapter_config import AdapterConfig


DEFAULT_FLOAT_DTYPE = np.float32
DEFAULT_STATS_ROOT = Path(__file__).resolve().parent / "saved" / "statistics"

DEFAULT_FULL_SHAPES: dict[str, tuple[int, int]] = {
    "3km": (92, 68),
    "12km": (23, 17),
}

def parse_iso_timestamp(timestamp_str: str) -> datetime:
    if not isinstance(timestamp_str, str):
        raise TypeError(f"Expected timestamp string, got {type(timestamp_str)}")
    return datetime.fromisoformat(timestamp_str)


def normalize_timestamp_key(timestamp: Any) -> str:
    """
    Normalize timestamps to canonical ISO format with a 'T' separator.
    """
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    if isinstance(timestamp, str):
        return parse_iso_timestamp(timestamp).isoformat()
    raise TypeError(f"Unsupported timestamp type for normalization: {type(timestamp)}")


def sample_entry_get(sample: Any, key: str, default: Any = None) -> Any:
    """
    Access a sample-index entry field from either a dataclass-like object or a dict.
    """
    if hasattr(sample, key):
        return getattr(sample, key)
    if isinstance(sample, dict):
        return sample.get(key, default)
    return default


def loaded_field_get(field: Any, key: str, default: Any = None) -> Any:
    """
    Access a loaded field attribute from either an object-like return value or a dict.
    """
    if hasattr(field, key):
        return getattr(field, key)
    if isinstance(field, dict):
        return field.get(key, default)
    return default

def build_doy_sincos(
    timestamp_str: str,
    *,
    dtype: np.dtype = DEFAULT_FLOAT_DTYPE,  # type: ignore
) -> np.ndarray:
    dt = parse_iso_timestamp(timestamp_str)
    doy = int(dt.timetuple().tm_yday)
    theta = 2.0 * math.pi * float(doy - 1) / 365.0
    return np.asarray([math.sin(theta), math.cos(theta)], dtype=dtype)

def build_date_metadata(timestamp_str: str) -> dict[str, Any]:
    dt = parse_iso_timestamp(timestamp_str)
    return {
        #"date": dt.strftime("%Y%m%d"),
        "date": dt.strftime("%Y%m%d%H"),
        "day_of_year": int(dt.timetuple().tm_yday),
        "doy_sin_cos": build_doy_sincos(timestamp_str).tolist(),
    }

class NorCPDataset(Dataset):
    """
    NorCP dataset implementation for STRIDE.
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        self.cfg = cfg
        self.root_dir = Path(cfg.root_dir)
        self.scenario_name = self._get_required_attr("scenario_name")
        self.target_variable = self._get_target_variable()
        self.target_variables = [self.target_variable]
        self.dynamic_variables = list(self._get_dynamic_variables())
        self.static_variables = list(self._get_static_variables())

        self.target_spatial_tag = getattr(cfg, "target_spatial_tag", "3km")
        self.dynamic_spatial_tag = getattr(cfg, "dynamic_spatial_tag", "12km")
        self.temporal_tag = getattr(cfg, "temporal_tag", "6hr")
        self.domain_tag = getattr(cfg, "domain_tag", "full_domain")
        self.spatial_shuffle = bool(getattr(cfg, "spatial_shuffle", False))
        self.shuffle_seed = getattr(cfg, "shuffle_seed", None)
        self._rng = np.random.default_rng(self.shuffle_seed)
        self.split_name = getattr(cfg, "split_name", "train")
        self.statistics_split_map = getattr(cfg, "statistics_split_map", None)
        self.stats_split_name = self._resolve_stats_split_name()
        self.split_manifest_path = Path(self._get_required_attr("split_manifest_path"))
        self.apply_transforms = bool(getattr(cfg, "apply_transforms", True))

        self.target_source = getattr(cfg, "target_source", "NORCP_HR")
        self.dynamic_source = getattr(cfg, "dynamic_source", "NORCP_LR")
        self.static_source = getattr(cfg, "static_source", "NORCP_STATIC")
        self.static_allow_missing = bool(getattr(cfg, "static_allow_missing", True))

        self.target_time_offsets = dict(getattr(cfg, "target_time_offsets", {"prcp": -3.0}))
        self.dynamic_time_offsets = dict(getattr(cfg, "dynamic_time_offsets", {"prcp": -3.0}))

        self.hr_crop_spec = self._build_crop_spec(getattr(cfg, "crop", None))
        self.lr_crop_spec = (
            None
            if self.hr_crop_spec is None
            else derive_lr_crop_from_hr_crop(self.hr_crop_spec)
        )
        self.target_full_shape = self._resolve_full_shape(
            spatial_tag=self.target_spatial_tag,
            cfg_value=getattr(cfg, "target_full_shape", None),
        )
        self.dynamic_full_shape = self._resolve_full_shape(
            spatial_tag=self.dynamic_spatial_tag,
            cfg_value=getattr(cfg, "dynamic_full_shape", None),
        )

        self.split_manifest = load_split_manifest(self.split_manifest_path)
        self.timestamps = [
            normalize_timestamp_key(timestamp)
            for timestamp in get_split_timestamps(self.split_manifest, self.split_name)
        ]

        #print('NorCPDataset: self.root_dir:', self.root_dir)
        #print('NorCPDataset: self.scenario_name:', self.scenario_name)
        #print('NorCPDataset: self.target_variables:', self.target_variables)
        #print('NorCPDataset: self.dynamic_variables:', self.dynamic_variables)
        #print('NorCPDataset: self.temporal_tag:', self.temporal_tag)
        #print('NorCPDataset: self.target_time_offsets:', self.target_time_offsets)
        self.sample_index = build_norcp_sample_index(
            root_dir=self.root_dir,
            scenario_name=self.scenario_name,
            target_variables=self.target_variables,
            dynamic_variables=self.dynamic_variables,
            target_spatial_tag=self.target_spatial_tag,
            dynamic_spatial_tag=self.dynamic_spatial_tag,
            temporal_tag=self.temporal_tag,
            target_time_offsets=self.target_time_offsets,
            dynamic_time_offsets=self.dynamic_time_offsets,
        )
        #print('NorCPDataset: get_split_timestamps(self.split_manifest, self.split_name)', get_split_timestamps(self.split_manifest, self.split_name))
        #print('NorCPDataset: self.sample_index', self.sample_index)
        #print('NorCPDataset: sample_entry_get(sample, "timestamp")', sample_entry_get(sample, "timestamp"))
        self._sample_lookup = {
            normalize_timestamp_key(sample_entry_get(sample, "timestamp")): sample
            for sample in self.sample_index
        }
        self._validate_timestamps_exist()
        self.static_file_paths = self._build_static_file_paths()

        self.stats_root = self._resolve_stats_root()

        self.target_transforms = self._build_transforms(
            variables=self.target_variables,
            source=self.target_source,
        )
        self.dynamic_transforms = self._build_transforms(
            variables=self.dynamic_variables,
            source=self.dynamic_source,
        )
        self.static_transforms = self._build_transforms(
            variables=self.static_variables,
            source=self.static_source,
        )

    def _resolve_stats_split_name(self) -> str:
        """
        Resolve which split's statistics should be used for normalization.
        Dataset split and statistics split are intentionally decoupled:
        - split_name controls which samples are loaded
        - stats_split_name controls which saved normalization statistics are used

        Current default policy is to use training statistics for all splits unless
        an explicit statistics_split_map is provided by the compiled config.
        """
        raw_map = self.statistics_split_map
        if isinstance(raw_map, dict):
            selected = raw_map.get(self.split_name, None)
            if selected is not None:
                return str(selected)
        return "train"


    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, index: int) -> dict[str, Any]:
        timestamp = self.timestamps[index]
        sample = self._sample_lookup[timestamp]

        hr_crop_spec, lr_crop_spec = self._resolve_crop_specs_for_sample()

        target_arr, target_field_meta = self._load_temporal_stack(
            file_mapping=sample_entry_get(sample, "target_files", {}),
            variable_order=self.target_variables,
            source=self.target_source,
            aligned_timestamp=timestamp,
            time_offsets=self.target_time_offsets,
            crop_spec=self.hr_crop_spec,
        )
        dynamic_arr, dynamic_field_meta = self._load_temporal_stack(
            file_mapping=sample_entry_get(sample, "dynamic_files", {}),
            variable_order=self.dynamic_variables,
            source=self.dynamic_source,
            aligned_timestamp=timestamp,
            time_offsets=self.dynamic_time_offsets,
            crop_spec=self.lr_crop_spec,
        )
        static_arr, static_field_meta = self._load_static_stack(
            file_mapping=self.static_file_paths, # type: ignore
            variable_order=self.static_variables,
            source=self.static_source,
            crop_spec=self.hr_crop_spec,
        )

        if self.apply_transforms:
            target_arr = self._apply_channelwise_transforms(
                target_arr,
                variable_names=self.target_variables,
                transforms=self.target_transforms,
            )
            dynamic_arr = self._apply_channelwise_transforms(
                dynamic_arr,
                variable_names=self.dynamic_variables,
                transforms=self.dynamic_transforms,
            )
            if static_arr is not None:
                static_arr = self._apply_channelwise_transforms(
                    static_arr,
                    variable_names=self.static_variables,
                    transforms=self.static_transforms,
                )

        time_features_arr = build_doy_sincos(timestamp)
        cond_coord = self._build_region_info(
            target_shape=(int(target_arr.shape[-2]), int(target_arr.shape[-1])),
            target_crop_spec=hr_crop_spec,
        )
        meta = self._build_meta(
            timestamp=timestamp,
            sample=sample, # type: ignore
            target_field_meta=target_field_meta,
            dynamic_field_meta=dynamic_field_meta,
            static_field_meta=static_field_meta,
            cond_coord=cond_coord,
        )

        target_tensor = torch.from_numpy(np.ascontiguousarray(target_arr)).to(torch.float32)
        cond_dynamic_tensor = torch.from_numpy(np.ascontiguousarray(dynamic_arr)).to(torch.float32)
        cond_static_tensor = (
            torch.from_numpy(np.ascontiguousarray(static_arr)).to(torch.float32)
            if static_arr is not None
            else None
        )
        time_features_tensor = torch.from_numpy(np.ascontiguousarray(time_features_arr)).to(torch.float32)

        return {
            "target": target_tensor,
            "cond_dynamic": cond_dynamic_tensor,
            "cond_static": cond_static_tensor,
            "cond_coord": cond_coord,
            "meta": meta,
            "time_features": time_features_tensor,
        }

    def _get_required_attr(self, name: str) -> Any:
        if not hasattr(self.cfg, name):
            raise AttributeError(f"NorCP adapter config is missing required attribute '{name}'")
        return getattr(self.cfg, name)

    def _get_target_variable(self) -> str:
        if hasattr(self.cfg, "target_variable"):
            return str(getattr(self.cfg, "target_variable"))
        if hasattr(self.cfg, "target_variables"):
            values = list(getattr(self.cfg, "target_variables"))
            if len(values) != 1:
                raise ValueError(
                    "NorCP adapter currently supports exactly one target variable. "
                    f"Got: {values}"
                )
            return str(values[0])
        raise AttributeError(
            "NorCP adapter config must define either 'target_variable' or a single-item 'target_variables'"
        )

    def _get_dynamic_variables(self) -> list[str]:
        if hasattr(self.cfg, "dynamic_variables"):
            return list(getattr(self.cfg, "dynamic_variables"))
        return []

    def _get_static_variables(self) -> list[str]:
        if hasattr(self.cfg, "static_variables"):
            return list(getattr(self.cfg, "static_variables"))
        return []

    def _resolve_stats_root(self) -> Path:
        if hasattr(self.cfg, "stats_root") and getattr(self.cfg, "stats_root") is not None:
            return Path(getattr(self.cfg, "stats_root"))

        split_manifest_tag = Path(self.split_manifest_path).stem
        return DEFAULT_STATS_ROOT / split_manifest_tag

    def _resolve_full_shape(
        self,
        *,
        spatial_tag: str,
        cfg_value: Any,
    ) -> tuple[int, int]:
        if cfg_value is not None:
            if isinstance(cfg_value, (list, tuple)) and len(cfg_value) == 2:
                return (int(cfg_value[0]), int(cfg_value[1]))
            raise ValueError(
                f"Expected full shape for spatial_tag='{spatial_tag}' to be a length-2 tuple/list, got: {cfg_value}"
            )
        if spatial_tag in DEFAULT_FULL_SHAPES:
            return DEFAULT_FULL_SHAPES[spatial_tag]
        raise KeyError(
            f"No default full shape known for spatial_tag='{spatial_tag}'. Please provide it in config."
        )

    def _resolve_crop_specs_for_sample(self) -> tuple[CropSpec | None, CropSpec | None]:
        if self.hr_crop_spec is None:
            return None, None

        if not self.spatial_shuffle:
            return self.hr_crop_spec, self.lr_crop_spec

        max_top = self.target_full_shape[0] - self.hr_crop_spec.height
        max_left = self.target_full_shape[1] - self.hr_crop_spec.width
        if max_top < 0 or max_left < 0:
            raise ValueError(
                "Configured crop is larger than the full target domain: "
                f"crop={self.hr_crop_spec}, full_shape={self.target_full_shape}"
            )

        top = int(self._rng.integers(0, max_top + 1))
        left = int(self._rng.integers(0, max_left + 1))
        hr_crop = CropSpec(
            top=top,
            left=left,
            height=self.hr_crop_spec.height,
            width=self.hr_crop_spec.width,
        )
        lr_crop = derive_lr_crop_from_hr_crop(hr_crop)
        return hr_crop, lr_crop

    def _validate_timestamps_exist(self) -> None:
        missing = [timestamp for timestamp in self.timestamps if timestamp not in self._sample_lookup]
        #print('_validate_timestamps_exist: self.timestamps:', self.timestamps[:10])
        #print('_validate_timestamps_exist: self._sample_lookup:', self._sample_lookup)
        if missing:
            raise ValueError(
                f"Some split timestamps are missing from the NorCP sample index: {missing[:10]}"
            )

    def _build_static_file_paths(self) -> dict[str, str]:
        """
        Resolve static-file paths once at adapter initialization.

        NorCP static variables are not part of the temporal sample index. They are
        fixed fields that should be attached to every returned sample when
        available, mirroring the DANRA-style adapter structure.
        """
        if not self.static_variables:
            return {}

        resolved: dict[str, str] = {}
        for variable in self.static_variables:
            raw_name = "orog" if variable in {"topo", "orog", "height", "elevation"} else variable
            candidate = (
                self.root_dir
                / self.scenario_name
                / self.target_spatial_tag
                / "fx"
                / raw_name
                / f"{raw_name}_{self.target_spatial_tag}_fx.nc"
            )

            if candidate.exists():
                resolved[variable] = str(candidate)
                continue

            if self.static_allow_missing:
                continue

            raise FileNotFoundError(
                "Required NorCP static file does not exist: "
                f"variable='{variable}', path='{candidate}'"
            )

        return resolved

    def _build_transforms(
        self,
        *,
        variables: list[str],
        source: str,
    ) -> dict[str, BaseTransform]:
        transforms: dict[str, BaseTransform] = {}
        for variable in variables:
            transform_name = get_default_transform(variable)
            if not self.apply_transforms:
                transforms[variable] = build_transform(
                    variable=variable,
                    source=source,
                    transform_name="identity",
                    stats=None,
                )
                continue

            stats_path = build_stats_output_path(
                split_name=self.stats_split_name,
                domain_tag=self.domain_tag,
                scenario_name=self.scenario_name,
                variable=variable,
                source=source,
                transform_name=transform_name,
                root_dir=self.stats_root,
            )
            stats = load_transform_stats(stats_path)
            transforms[variable] = build_transform(
                variable=variable,
                source=source,
                transform_name=transform_name,
                stats=stats,
            )
        return transforms

    def _load_temporal_stack(
        self,
        *,
        file_mapping: dict[str, str | Path],
        variable_order: list[str],
        source: str,
        aligned_timestamp: str,
        time_offsets: dict[str, float],
        crop_spec: CropSpec | None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        arrays: list[np.ndarray] = []
        metadata_list: list[dict[str, Any]] = []

        for variable in variable_order:
            if variable not in file_mapping:
                raise KeyError(
                    f"Variable '{variable}' not found in file mapping for source '{source}'"
                )
            field = load_field_at_timestamp(
                file_path=file_mapping[variable],
                variable=variable,
                source=source,
                timestamp=parse_iso_timestamp(aligned_timestamp),
                file_time_offset_hours=float(time_offsets.get(variable, 0.0)),
            )
            field_array = loaded_field_get(field, "array")
            if field_array is None:
                raise AttributeError(
                    f"Loaded field for variable '{variable}' does not expose an 'array' attribute"
                )
            array = np.asarray(field_array, dtype=np.float32)
            array = self._maybe_crop_2d_field(array, crop_spec)
            arrays.append(array)

            field_metadata = loaded_field_get(field, "metadata", None)
            if field_metadata is None:
                field_metadata = {
                    "variable": variable,
                    "source": source,
                    "timestamp": aligned_timestamp,
                    "path": str(file_mapping[variable]),
                    "shape": tuple(array.shape),
                    "dtype": str(array.dtype),
                    "min": float(np.nanmin(array)),
                    "max": float(np.nanmax(array)),
                }
            metadata_list.append(dict(field_metadata))

        return np.stack(arrays, axis=0).astype(np.float32, copy=False), metadata_list

    def _load_static_stack(
        self,
        *,
        file_mapping: dict[str, str | Path],
        variable_order: list[str],
        source: str,
        crop_spec: CropSpec | None,
    ) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
        if not variable_order:
            return None, []

        arrays: list[np.ndarray] = []
        metadata_list: list[dict[str, Any]] = []
        for variable in variable_order:
            if variable not in file_mapping:
                continue
            field = load_static_field(
                file_path=file_mapping[variable],
                variable=variable,
                source=source,
            )
            field_array = loaded_field_get(field, "array")
            if field_array is None:
                raise AttributeError(
                    f"Loaded static field for variable '{variable}' does not expose an 'array' attribute"
                )
            array = np.asarray(field_array, dtype=np.float32)
            array = self._maybe_crop_2d_field(array, crop_spec)
            arrays.append(array)

            field_metadata = loaded_field_get(field, "metadata", None)
            if field_metadata is None:
                field_metadata = {
                    "variable": variable,
                    "source": source,
                    "timestamp": None,
                    "path": str(file_mapping[variable]),
                    "shape": tuple(array.shape),
                    "dtype": str(array.dtype),
                    "min": float(np.nanmin(array)),
                    "max": float(np.nanmax(array)),
                }
            metadata_list.append(dict(field_metadata))

        if not arrays:
            return None, []
        return np.stack(arrays, axis=0).astype(np.float32, copy=False), metadata_list

    @staticmethod
    def _build_crop_spec(crop: tuple[int, int, int, int] | None) -> CropSpec | None:
        if crop is None:
            return None
        top, left, height, width = crop
        return CropSpec(top=int(top), left=int(left), height=int(height), width=int(width))

    @staticmethod
    def _maybe_crop_2d_field(array: np.ndarray, crop_spec: CropSpec | None) -> np.ndarray:
        if crop_spec is None:
            return array
        stacked = np.expand_dims(array, axis=0)
        cropped = maybe_crop_3d(stacked, crop_spec)
        return np.asarray(cropped[0], dtype=np.float32)

    @staticmethod
    def _apply_channelwise_transforms(
        array: np.ndarray,
        variable_names: list[str],
        transforms: dict[str, BaseTransform],
    ) -> np.ndarray:
        if array.shape[0] != len(variable_names):
            raise ValueError(
                f"Channel count {array.shape[0]} does not match variable list {variable_names}"
            )

        transformed_channels: list[np.ndarray] = []
        for idx, variable_name in enumerate(variable_names):
            transformed = transforms[variable_name].forward(array[idx])
            transformed_channels.append(np.asarray(transformed, dtype=np.float32))
        return np.stack(transformed_channels, axis=0)

    def _build_region_info(
        self,
        *,
        target_shape: tuple[int, int],
        target_crop_spec: CropSpec | None,
    ) -> dict[str, Any]:
        if target_crop_spec is None:
            crop_anchor_y = 0
            crop_anchor_x = 0
            crop_height = int(target_shape[0])
            crop_width = int(target_shape[1])
        else:
            crop_anchor_y = int(target_crop_spec.top)
            crop_anchor_x = int(target_crop_spec.left)
            crop_height = int(target_crop_spec.height)
            crop_width = int(target_crop_spec.width)

        return {
            "full_height": int(self.target_full_shape[0]),
            "full_width": int(self.target_full_shape[1]),
            "crop_anchor_y": crop_anchor_y,
            "crop_anchor_x": crop_anchor_x,
            "crop_height": crop_height,
            "crop_width": crop_width,
        }

    def _build_meta(
        self,
        *,
        timestamp: str,
        sample: dict[str, Any],
        target_field_meta: list[dict[str, Any]],
        dynamic_field_meta: list[dict[str, Any]],
        static_field_meta: list[dict[str, Any]],
        cond_coord: dict[str, Any],
    ) -> dict[str, Any]:
        date_meta = build_date_metadata(timestamp)
        return {
            "timestamp": timestamp,
            "date": date_meta["date"],
            "day_of_year": date_meta["day_of_year"],
            "doy_sin_cos": date_meta["doy_sin_cos"],
            "scenario_name": self.scenario_name,
            "split_name": self.split_name,
            "stats_split_name": self.stats_split_name,            
            "split_manifest_path": str(self.split_manifest_path),
            "domain_tag": self.domain_tag,
            "target_var": self.target_variable,
            "target_variables": list(self.target_variables),
            "cond_dynamic_vars": list(self.dynamic_variables),
            "cond_static_vars": list(self.static_variables),
            "target_source": self.target_source,
            "dynamic_source": self.dynamic_source,
            "static_source": self.static_source,
            "static_allow_missing": self.static_allow_missing,
            "apply_transforms": self.apply_transforms,
            "spatial_shuffle": self.spatial_shuffle,
            "sample_index_metadata": dict(sample_entry_get(sample, "metadata", {}) or {}),
            "region_info": cond_coord,
            "field_metadata": {
                "target": target_field_meta,
                "dynamic": dynamic_field_meta,
                "static": static_field_meta,
            },
            "transform_metadata": {
                "target": {
                    name: transform.metadata().to_dict()
                    for name, transform in self.target_transforms.items()
                },
                "dynamic": {
                    name: transform.metadata().to_dict()
                    for name, transform in self.dynamic_transforms.items()
                },
                "static": {
                    name: transform.metadata().to_dict()
                    for name, transform in self.static_transforms.items()
                },
            },
        }


class NorCPAdapter:
    """
    Thin adapter wrapper around the NorCP dataset implementation.
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        self.cfg = cfg

    def build_dataset(self) -> NorCPDataset:
        return NorCPDataset(self.cfg)

    def build_datasets(self) -> dict[str, NorCPDataset]:
        split_names = ("train", "val", "test")
        datasets: dict[str, NorCPDataset] = {}
        for split_name in split_names:
            split_cfg = self._clone_cfg_with_split(split_name)
            #print('NorCPAdapter: split_cfg', split_cfg)
            datasets[split_name] = NorCPDataset(split_cfg)
        return datasets

    def _clone_cfg_with_split(self, split_name: str) -> AdapterConfig:
        cfg_kwargs = dict(vars(self.cfg))
        cfg_kwargs["split_name"] = split_name
        return AdapterConfig(**cfg_kwargs)
