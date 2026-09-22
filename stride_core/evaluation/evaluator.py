"""
Main evaluation orchestration

Responsibilities
----------------
- Initialize evaluation configuration
- Build evaluation dataset/index
- Load generated products and references
- Dispatch metric families
- Dispatch plotting
- Save outputs

The evaluator intentionally separates orchestration from metric
implementations. Metric implementations will live in dedicated modules
(e.g. probabilistic_metrics.py, spatial_metrics.py, etc.).
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any
import json
import time
import logging

from stride_core.evaluation.evaluation_config import EvaluationRunConfig
from stride_core.evaluation.data_loading import EvaluationDataLoader
from stride_core.evaluation.targets import (
    get_climatology_product,
    get_probabilistic_product,
    get_spatial_product,
    get_temporal_product,
    summarize_target_routing,
)
from stride_core.evaluation.summary import summary_metrics_as_dicts
from stride_core.evaluation.summary_writer import (
    write_summary_csv,
    write_summary_markdown,
)
from stride_core.evaluation.metrics.probabilistic import (
    compute_crps,
    compute_crps_decomposition,
    compute_pit_histogram,
    compute_rank_histogram,
    compute_reliability,
    compute_spread_skill,
)
from stride_core.evaluation.metrics.spatial import (
    compute_iss,
    compute_psd,
    compute_psd_slope,
    compute_sal,
)

from stride_core.evaluation.metrics.climatology import (
    compute_annual_precipitation_sum,
    compute_extremes,
    compute_histogram_comparison,
    compute_pixel_value_distribution,
    compute_qq_data,
    compute_seasonal_accumulations,
    compute_wet_day_frequency,
)

from stride_core.evaluation.metrics.temporal import (
    compute_dry_spell_lengths,
    compute_lag_autocorrelation,
    compute_wet_spell_lengths,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    High‑level evaluation driver.

    Flow
    ----
    1. Parse evaluation configuration
    2. Initialize data loader
    3. Iterate over evaluation cases
    4. Dispatch enabled metric families
    5. Save results
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, cfg: EvaluationRunConfig):
        self.cfg = cfg

        # Output directory
        self.output_dir: Path = cfg.outputs.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data loader
        self.loader = EvaluationDataLoader(cfg)

        # Storage for computed metrics
        self.metrics: dict[str, Any] = {
            "target_routing": summarize_target_routing(),
            "summary": [],
            "probabilistic": {},
            "spatial": {},
            "climatology": {},
            "temporal": {},
            "sigma_star": {},
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the evaluation pipeline."""

        start_time = time.time()

        self._print_run_summary()

        cases = list(self.loader.iter_cases())
        print('run:cases', cases)
        dates = [c.date for c in cases]
        print(f"n_cases:    {len(cases)}")
        print(f"date range: {min(dates)} → {max(dates)}")
        print(f"first 5:    {dates[:5]}")

        for case in cases[:1]:
            for prod_name, prod in case.products.items():
                print(f"\nProduct: {prod_name}")
                for arr_name, arr in prod.arrays.items():
                    print(f"  {arr_name:30s}  shape={arr.shape}  dtype={arr.dtype}")

        if len(cases) == 0:
            raise RuntimeError("No evaluation cases found.")

        # Dispatch metric families
        if self._any_probabilistic_enabled():
            self.run_probabilistic_metrics(cases)

        if self._any_spatial_enabled():
            self.run_spatial_metrics(cases)

        if self._any_climatology_enabled():
            self.run_climatology_metrics(cases)

        if self._any_temporal_enabled():
            self.run_temporal_metrics(cases)

        if self.cfg.sigma_star.enabled:
            self.run_sigma_star_analysis(cases)

        self._build_summary_metrics()
        self._run_plotting()

        # Save results
        self._save_metrics()

        elapsed = time.time() - start_time
        logger.info(f"\nEvaluation completed in {elapsed:.2f} seconds")
    def _run_plotting(self) -> None:
        """
        Dispatch family-specific plotting modules when enabled.

        Plotting remains optional and is intentionally treated as a second pass
        over the already-computed metric outputs.
        """
        if not self._plotting_enabled():
            logger.info("Plotting disabled; skipping figure generation.")
            return

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        for family in ("probabilistic", "spatial", "climatology", "temporal"):
            if not self._plotting_family_enabled(family):
                continue

            family_metrics = self.metrics.get(family, {})
            if not isinstance(family_metrics, dict) or len(family_metrics) == 0:
                logger.info(
                    "No %s metrics available for plotting; skipping.",
                    family,
                )
                continue

            self._run_family_plotter(family=family, figures_dir=figures_dir)

    def _run_family_plotter(self, *, family: str, figures_dir: Path) -> None:
        """
        Import and execute a family plotter if one exists.

        Expected convention
        -------------------
        Module:
            stride_core.evaluation.plots.plot_<family>

        Function names (first match wins):
            plot_<family>
            run_<family>_plots
        """
        module_name = f"stride_core.evaluation.plots.plot_{family}"

        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            logger.info(
                "No plotting module found for %s (%s); skipping.",
                family,
                module_name,
            )
            return
        except Exception as exc:
            logger.warning(
                "Failed to import plotting module for %s: %s",
                family,
                exc,
            )
            return

        candidate_names = (
            f"plot_{family}",
            f"run_{family}_plots",
        )

        plot_fn = None
        for name in candidate_names:
            fn = getattr(module, name, None)
            if callable(fn):
                plot_fn = fn
                break

        if plot_fn is None:
            logger.info(
                "Plotting module %s does not expose a recognized entrypoint; skipping.",
                module_name,
            )
            return

        family_dir = figures_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)

        try:
            plot_fn(
                metrics=self.metrics,
                output_dir=family_dir,
                cfg=self.cfg,
            )
            logger.info("Saved %s plots to %s", family, family_dir)
        except TypeError:
            # Backward-compatible fallback for simpler plotter signatures.
            plot_fn(self.metrics, family_dir, self.cfg)
            logger.info("Saved %s plots to %s", family, family_dir)
        except Exception as exc:
            logger.warning("Failed while generating %s plots: %s", family, exc)

    def _plotting_enabled(self) -> bool:
        plotting_cfg = getattr(self.cfg, "plotting", None)
        if plotting_cfg is None:
            return False
        return bool(getattr(plotting_cfg, "enabled", False))

    def _plotting_family_enabled(self, family: str) -> bool:
        plotting_cfg = getattr(self.cfg, "plotting", None)
        if plotting_cfg is None:
            return False

        families = getattr(plotting_cfg, "families", None)
        if families in (None, {}):
            return True
        if isinstance(families, dict):
            return bool(families.get(family, True))

        return True

    # ------------------------------------------------------------------
    # Metric family dispatch
    # ------------------------------------------------------------------

    def run_probabilistic_metrics(self, cases) -> None:
        """Run ensemble probabilistic verification metrics."""

        logger.info("\nRunning probabilistic metrics...")

        case_level: dict[str, Any] = {}
        forecast_list = []
        target_list = []

        for case in cases:
            metric_cache: dict[str, tuple[Any, Any]] = {}
            primary_forecast, primary_target = self._get_metric_arrays(
                case,
                family="probabilistic",
                metric="crps",
                cache=metric_cache,
            )

            forecast_list.append(primary_forecast)
            target_list.append(primary_target)

            case_entry: dict[str, Any] = {
                "forecast_shape": tuple(primary_forecast.shape),
                "target_shape": tuple(primary_target.shape),
                "metric_products": {},
            }

            if self.cfg.probabilistic.crps:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="probabilistic",
                    metric="crps",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["crps"] = get_probabilistic_product("crps")
                case_entry["crps"] = self._to_serializable(
                    compute_crps(forecast, target)
                )

            if self.cfg.probabilistic.crps_decomposition:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="probabilistic",
                    metric="crps_decomposition",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["crps_decomposition"] = get_probabilistic_product("crps_decomposition")
                case_entry["crps_decomposition"] = self._to_serializable(
                    compute_crps_decomposition(forecast, target)
                )

            if self.cfg.probabilistic.spread_skill:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="probabilistic",
                    metric="spread_skill",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["spread_skill"] = get_probabilistic_product("spread_skill")
                case_entry["spread_skill"] = self._to_serializable(
                    compute_spread_skill(forecast, target)
                )

            if self.cfg.probabilistic.pit_histogram:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="probabilistic",
                    metric="pit_histogram",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["pit_histogram"] = get_probabilistic_product("pit_histogram")
                case_entry["pit_histogram"] = self._to_serializable(
                    compute_pit_histogram(forecast, target)
                )

            if self.cfg.probabilistic.rank_histogram:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="probabilistic",
                    metric="rank_histogram",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["rank_histogram"] = get_probabilistic_product("rank_histogram")
                case_entry["rank_histogram"] = self._to_serializable(
                    compute_rank_histogram(forecast, target)
                )

            if self.cfg.probabilistic.reliability_diagram:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="probabilistic",
                    metric="reliability_diagram",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["reliability_diagram"] = get_probabilistic_product("reliability_diagram")
                case_entry["reliability_diagram"] = self._to_serializable(
                    compute_reliability(
                        forecast,
                        target,
                        thresholds_mm=self.cfg.thresholds.reliability_thresholds_mm,
                    )
                )

            case_level[case.case_id] = case_entry

        family_results: dict[str, Any] = {
            "case_level": case_level,
            "num_cases": len(cases),
        }

        if len(forecast_list) > 0:
            family_results["aggregate"] = {}
            stacked_forecast = self._stack_case_forecasts(forecast_list)
            stacked_target = self._stack_case_targets(target_list)

            if self.cfg.probabilistic.crps:
                family_results["aggregate"]["crps"] = self._to_serializable(
                    compute_crps(stacked_forecast, stacked_target)
                )

            if self.cfg.probabilistic.crps_decomposition:
                family_results["aggregate"]["crps_decomposition"] = self._to_serializable(
                    compute_crps_decomposition(stacked_forecast, stacked_target)
                )

            if self.cfg.probabilistic.spread_skill:
                family_results["aggregate"]["spread_skill"] = self._to_serializable(
                    compute_spread_skill(stacked_forecast, stacked_target)
                )

            if self.cfg.probabilistic.pit_histogram:
                family_results["aggregate"]["pit_histogram"] = self._to_serializable(
                    compute_pit_histogram(stacked_forecast, stacked_target)
                )

            if self.cfg.probabilistic.rank_histogram:
                family_results["aggregate"]["rank_histogram"] = self._to_serializable(
                    compute_rank_histogram(stacked_forecast, stacked_target)
                )

            if self.cfg.probabilistic.reliability_diagram:
                family_results["aggregate"]["reliability_diagram"] = self._to_serializable(
                    compute_reliability(
                        stacked_forecast,
                        stacked_target,
                        thresholds_mm=self.cfg.thresholds.reliability_thresholds_mm,
                    )
                )

        self.metrics["probabilistic"] = family_results

    def run_spatial_metrics(self, cases) -> None:
        """Run spatial structure metrics."""

        logger.info("\nRunning spatial metrics...")

        case_level: dict[str, Any] = {}

        psd_forecast_list = []
        psd_target_list = []
        iss_forecast_list = []
        iss_target_list = []
        sal_forecast_list = []
        sal_target_list = []

        for case in cases:
            metric_cache: dict[str, tuple[Any, Any]] = {}
            case_entry: dict[str, Any] = {
                "metric_products": {},
            }

            if self.cfg.spatial.psd:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="spatial",
                    metric="psd",
                    cache=metric_cache,
                )
                case_entry["forecast_shape"] = tuple(forecast.shape)
                case_entry["target_shape"] = tuple(target.shape)
                case_entry["metric_products"]["psd"] = get_spatial_product("psd")
                case_entry["psd"] = self._to_serializable(
                    compute_psd(forecast, target)
                )
                psd_forecast_list.append(forecast)
                psd_target_list.append(target)

            if self.cfg.spatial.psd_slope:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="spatial",
                    metric="psd_slope",
                    cache=metric_cache,
                )
                case_entry.setdefault("forecast_shape", tuple(forecast.shape))
                case_entry.setdefault("target_shape", tuple(target.shape))
                case_entry["metric_products"]["psd_slope"] = get_spatial_product("psd_slope")
                case_entry["psd_slope"] = self._to_serializable(
                    compute_psd_slope(
                        forecast,
                        target,
                        fit_range_km=self.cfg.scales.psd_slope_fit_range_km,
                    )
                )

            if self.cfg.spatial.iss:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="spatial",
                    metric="iss",
                    cache=metric_cache,
                )
                case_entry.setdefault("forecast_shape", tuple(forecast.shape))
                case_entry.setdefault("target_shape", tuple(target.shape))
                case_entry["metric_products"]["iss"] = get_spatial_product("iss")
                case_entry["iss"] = self._to_serializable(
                    compute_iss(
                        forecast,
                        target,
                        thresholds_mm=self.cfg.thresholds.iss_thresholds_mm,
                        scales_km=self.cfg.scales.iss_scales_km,
                    )
                )
                iss_forecast_list.append(forecast)
                iss_target_list.append(target)

            if self.cfg.spatial.sal:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="spatial",
                    metric="sal",
                    cache=metric_cache,
                )
                case_entry.setdefault("forecast_shape", tuple(forecast.shape))
                case_entry.setdefault("target_shape", tuple(target.shape))
                case_entry["metric_products"]["sal"] = get_spatial_product("sal")
                case_entry["sal"] = self._to_serializable(
                    compute_sal(
                        forecast,
                        target,
                        threshold_mm=self.cfg.thresholds.wet_day_threshold_mm,
                    )
                )
                sal_forecast_list.append(forecast)
                sal_target_list.append(target)

            case_level[case.case_id] = case_entry

        family_results: dict[str, Any] = {
            "case_level": case_level,
            "num_cases": len(cases),
        }

        aggregate: dict[str, Any] = {}

        if self.cfg.spatial.psd and len(psd_forecast_list) > 0:
            stacked_psd_forecast = self._stack_spatial_ensemble_forecasts(psd_forecast_list)
            stacked_psd_target = self._stack_spatial_targets(psd_target_list)
            aggregate["psd"] = self._to_serializable(
                compute_psd(stacked_psd_forecast, stacked_psd_target)
            )

        if self.cfg.spatial.psd_slope and len(psd_forecast_list) > 0:
            stacked_psd_forecast = self._stack_spatial_ensemble_forecasts(psd_forecast_list)
            stacked_psd_target = self._stack_spatial_targets(psd_target_list)
            aggregate["psd_slope"] = self._to_serializable(
                compute_psd_slope(
                    stacked_psd_forecast,
                    stacked_psd_target,
                    fit_range_km=self.cfg.scales.psd_slope_fit_range_km,
                )
            )

        if self.cfg.spatial.iss and len(iss_forecast_list) > 0:
            stacked_iss_forecast = self._stack_spatial_fields(iss_forecast_list)
            stacked_iss_target = self._stack_spatial_targets(iss_target_list)
            aggregate["iss"] = self._to_serializable(
                compute_iss(
                    stacked_iss_forecast,
                    stacked_iss_target,
                    thresholds_mm=self.cfg.thresholds.iss_thresholds_mm,
                    scales_km=self.cfg.scales.iss_scales_km,
                )
            )

        if self.cfg.spatial.sal and len(sal_forecast_list) > 0:
            stacked_sal_forecast = self._stack_spatial_fields(sal_forecast_list)
            stacked_sal_target = self._stack_spatial_targets(sal_target_list)
            aggregate["sal"] = self._to_serializable(
                compute_sal(
                    stacked_sal_forecast,
                    stacked_sal_target,
                    threshold_mm=self.cfg.thresholds.wet_day_threshold_mm,
                )
            )

        if len(aggregate) > 0:
            family_results["aggregate"] = aggregate

        self.metrics["spatial"] = family_results

    def run_climatology_metrics(self, cases) -> None:
        """Run climatological/statistical diagnostics."""

        logger.info("\nRunning climatology metrics...")

        case_level: dict[str, Any] = {}
        forecast_list = []
        target_list = []
        case_dates: list[str] = []

        for case in cases:
            metric_cache: dict[str, tuple[Any, Any]] = {}
            primary_forecast, primary_target = self._get_metric_arrays(
                case,
                family="climatology",
                metric="pixel_value_distribution",
                cache=metric_cache,
            )

            forecast_list.append(primary_forecast)
            target_list.append(primary_target)
            case_dates.append(case.date or case.case_id)

            case_entry: dict[str, Any] = {
                "forecast_shape": tuple(primary_forecast.shape),
                "target_shape": tuple(primary_target.shape),
                "metric_products": {},
            }

            if self.cfg.climatology.pixel_value_distribution:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="climatology",
                    metric="pixel_value_distribution",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["pixel_value_distribution"] = get_climatology_product("pixel_value_distribution")
                case_entry["pixel_value_distribution"] = self._to_serializable(
                    compute_pixel_value_distribution(forecast, target)
                )

            if self.cfg.climatology.histogram_comparison:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="climatology",
                    metric="histogram_comparison",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["histogram_comparison"] = get_climatology_product("histogram_comparison")
                case_entry["histogram_comparison"] = self._to_serializable(
                    compute_histogram_comparison(forecast, target)
                )

            if self.cfg.climatology.qq_plot:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="climatology",
                    metric="qq_plot",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["qq_plot"] = get_climatology_product("qq_plot")
                case_entry["qq_plot"] = self._to_serializable(
                    compute_qq_data(forecast, target)
                )

            if self.cfg.climatology.extremes:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="climatology",
                    metric="extremes",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["extremes"] = get_climatology_product("extremes")
                case_entry["extremes"] = self._to_serializable(
                    compute_extremes(
                        forecast,
                        target,
                        quantile_levels=self.cfg.thresholds.extreme_quantiles,
                    )
                )

            if self.cfg.climatology.wet_day_frequency:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="climatology",
                    metric="wet_day_frequency",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["wet_day_frequency"] = get_climatology_product("wet_day_frequency")
                case_entry["wet_day_frequency"] = self._to_serializable(
                    compute_wet_day_frequency(
                        forecast,
                        target,
                        threshold_mm=self.cfg.thresholds.wet_day_threshold_mm,
                    )
                )

            if self.cfg.climatology.annual_precipitation_sum:
                forecast, target = self._get_metric_arrays(
                    case,
                    family="climatology",
                    metric="annual_precipitation_sum",
                    cache=metric_cache,
                )
                case_entry["metric_products"]["annual_precipitation_sum"] = get_climatology_product("annual_precipitation_sum")
                case_entry["annual_precipitation_sum"] = self._to_serializable(
                    compute_annual_precipitation_sum(forecast, target)
                )

            case_level[case.case_id] = case_entry

        family_results: dict[str, Any] = {
            "case_level": case_level,
            "num_cases": len(cases),
        }

        if len(forecast_list) > 0:
            family_results["aggregate"] = {}
            stacked_forecast = self._stack_climatology_forecasts(forecast_list)
            stacked_target = self._stack_case_targets(target_list)

            if self.cfg.climatology.pixel_value_distribution:
                family_results["aggregate"]["pixel_value_distribution"] = self._to_serializable(
                    compute_pixel_value_distribution(stacked_forecast, stacked_target)
                )

            if self.cfg.climatology.histogram_comparison:
                family_results["aggregate"]["histogram_comparison"] = self._to_serializable(
                    compute_histogram_comparison(stacked_forecast, stacked_target)
                )

            if self.cfg.climatology.qq_plot:
                family_results["aggregate"]["qq_plot"] = self._to_serializable(
                    compute_qq_data(stacked_forecast, stacked_target)
                )

            if self.cfg.climatology.extremes:
                family_results["aggregate"]["extremes"] = self._to_serializable(
                    compute_extremes(
                        stacked_forecast,
                        stacked_target,
                        quantile_levels=self.cfg.thresholds.extreme_quantiles,
                    )
                )

            if self.cfg.climatology.wet_day_frequency:
                family_results["aggregate"]["wet_day_frequency"] = self._to_serializable(
                    compute_wet_day_frequency(
                        stacked_forecast,
                        stacked_target,
                        threshold_mm=self.cfg.thresholds.wet_day_threshold_mm,
                    )
                )

            if self.cfg.climatology.annual_precipitation_sum:
                family_results["aggregate"]["annual_precipitation_sum"] = self._to_serializable(
                    compute_annual_precipitation_sum(stacked_forecast, stacked_target)
                )

            if self.cfg.climatology.seasonal_accumulations:
                family_results["aggregate"]["seasonal_accumulations"] = self._to_serializable(
                    compute_seasonal_accumulations(
                        stacked_forecast,
                        stacked_target,
                        dates=case_dates,
                    )
                )

        self.metrics["climatology"] = family_results

    def run_temporal_metrics(self, cases) -> None:
        """Run temporal persistence / spell diagnostics."""

        logger.info("\nRunning temporal metrics...")

        case_level: dict[str, Any] = {}
        sequence_dates: list[str] = []
        forecast_sequence_list = []
        target_sequence_list = []

        for case in cases:
            metric_cache: dict[str, tuple[Any, Any]] = {}
            primary_forecast, primary_target = self._get_metric_arrays(
                case,
                family="temporal",
                metric="lag_autocorrelation",
                cache=metric_cache,
            )

            case_date = case.date or case.case_id
            sequence_dates.append(case_date)
            forecast_sequence_list.append(primary_forecast)
            target_sequence_list.append(primary_target)

            case_entry: dict[str, Any] = {
                "forecast_shape": tuple(primary_forecast.shape),
                "target_shape": tuple(primary_target.shape),
                "metric_products": {},
            }

            case_level[case.case_id] = case_entry

        family_results: dict[str, Any] = {
            "case_level": case_level,
            "num_cases": len(cases),
        }

        if len(forecast_sequence_list) > 0:
            family_results["aggregate"] = {}
            stacked_forecast = self._stack_temporal_forecasts(forecast_sequence_list)
            stacked_target = self._stack_temporal_targets(target_sequence_list)

            if self.cfg.temporal.lag_autocorrelation:
                family_results["aggregate"]["lag_autocorrelation"] = self._to_serializable(
                    compute_lag_autocorrelation(
                        stacked_forecast,
                        stacked_target,
                        lags=self.cfg.scales.autocorrelation_lags,
                    )
                )

            max_spell_length = stacked_forecast.shape[0]

            if self.cfg.temporal.wet_spell_length:
                family_results["aggregate"]["wet_spell_length"] = self._to_serializable(
                    compute_wet_spell_lengths(
                        stacked_forecast,
                        stacked_target,
                        threshold_mm=self.cfg.thresholds.spell_threshold_mm,
                        max_length=max_spell_length,
                    )
                )

            if self.cfg.temporal.dry_spell_length:
                family_results["aggregate"]["dry_spell_length"] = self._to_serializable(
                    compute_dry_spell_lengths(
                        stacked_forecast,
                        stacked_target,
                        threshold_mm=self.cfg.thresholds.spell_threshold_mm,
                        max_length=max_spell_length,
                    )
                )

        self.metrics["temporal"] = family_results

    def run_sigma_star_analysis(self, cases) -> None:
        """Run σ* analysis for scale-aware verification."""

        logger.info("\nRunning sigma-star analysis...")

        results = {
            "sigma_values": self.cfg.sigma_star.values,
            "num_cases": len(cases),
        }

        self.metrics["sigma_star"] = results

    def _stack_case_forecasts(self, forecasts: list[Any]):
        import numpy as np

        normalized = []
        for forecast in forecasts:
            arr = np.asarray(forecast)
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            elif arr.ndim != 4:
                raise ValueError(
                    "Each probabilistic forecast must have shape [M,H,W] or [B,M,H,W], "
                    f"got {tuple(arr.shape)}"
                )
            normalized.append(arr)
        return np.concatenate(normalized, axis=0)

    def _stack_case_targets(self, targets: list[Any]):
        import numpy as np

        normalized = []
        for target in targets:
            arr = np.asarray(target)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0, ...]
            elif arr.ndim != 3:
                raise ValueError(
                    "Each probabilistic target must have shape [H,W], [B,H,W], or [B,1,H,W], "
                    f"got {tuple(arr.shape)}"
                )
            normalized.append(arr)
        return np.concatenate(normalized, axis=0)

    def _stack_spatial_ensemble_forecasts(self, forecasts: list[Any]):
        import numpy as np

        normalized = []
        for forecast in forecasts:
            arr = np.asarray(forecast)
            if arr.ndim == 2:
                arr = arr[np.newaxis, np.newaxis, ...]
            elif arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 5 and arr.shape[2] == 1:
                arr = arr[:, :, 0, ...]
            elif arr.ndim != 4:
                raise ValueError(
                    "Each spatial ensemble forecast must have shape [H,W], [M,H,W], [B,M,H,W], or [B,M,1,H,W], "
                    f"got {tuple(arr.shape)}"
                )
            normalized.append(arr)
        return np.concatenate(normalized, axis=0)

    def _stack_spatial_fields(self, fields: list[Any]):
        import numpy as np

        normalized = []
        for field in fields:
            arr = np.asarray(field)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0, ...]
            elif arr.ndim != 3:
                raise ValueError(
                    "Each spatial field must have shape [H,W], [B,H,W], or [B,1,H,W], "
                    f"got {tuple(arr.shape)}"
                )
            normalized.append(arr)
        return np.concatenate(normalized, axis=0)

    def _stack_spatial_targets(self, targets: list[Any]):
        import numpy as np

        normalized = []
        for target in targets:
            arr = np.asarray(target)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0, ...]
            elif arr.ndim != 3:
                raise ValueError(
                    "Each spatial target must have shape [H,W], [B,H,W], or [B,1,H,W], "
                    f"got {tuple(arr.shape)}"
                )
            normalized.append(arr)
        return np.concatenate(normalized, axis=0)

    def _stack_climatology_forecasts(self, forecasts: list[Any]):
        import numpy as np

        normalized = []
        for forecast in forecasts:
            arr = np.asarray(forecast)
            if arr.ndim == 2:
                arr = arr[np.newaxis, np.newaxis, ...]
            elif arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            elif arr.ndim != 4:
                raise ValueError(
                    "Each climatology forecast must have shape [H,W], [M,H,W], or [B,M,H,W], "
                    f"got {tuple(arr.shape)}"
                )
            normalized.append(arr)
        return np.concatenate(normalized, axis=0)

    def _stack_temporal_forecasts(self, forecasts: list[Any]):
        import numpy as np

        normalized = []
        for forecast in forecasts:
            arr = np.asarray(forecast)
            if arr.ndim == 2:
                arr = arr[np.newaxis, np.newaxis, ...]
            elif arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            elif arr.ndim != 4:
                raise ValueError(
                    "Each temporal forecast must have shape [H,W], [M,H,W], or [T,M,H,W]-like sequence input, "
                    f"got {tuple(arr.shape)}"
                )
            normalized.append(arr)
        return np.concatenate(normalized, axis=0)

    def _stack_temporal_targets(self, targets: list[Any]):
        import numpy as np

        normalized = []
        for target in targets:
            arr = np.asarray(target)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0, ...]
            elif arr.ndim != 3:
                raise ValueError(
                    "Each temporal target must have shape [H,W], [T,H,W], or [T,1,H,W], "
                    f"got {tuple(arr.shape)}"
                )
            normalized.append(arr)
        return np.concatenate(normalized, axis=0)

    def _to_serializable(self, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, dict):
            return {key: self._to_serializable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        return value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _print_run_summary(self) -> None:
        logger.info("\n===========================")
        logger.info("STRIDE evaluation run")
        logger.info("===========================")
        logger.info(f"Run name:        {self.cfg.run_name}")
        logger.info(f"Cases found:     {len(self.loader)}")
        logger.info(f"Output dir:      {self.output_dir}")
        logger.info(f"Generation dir:  {self.cfg.paths.generation_output_dir}")

    def _save_metrics(self) -> None:

        if not self.cfg.outputs.save_metrics_json:
            return

        path = self.output_dir / "evaluation_metrics.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Saved metrics to {path}")

    # ------------------------------------------------------------------
    # Feature switches
    # ------------------------------------------------------------------

    def _any_probabilistic_enabled(self) -> bool:
        return any(vars(self.cfg.probabilistic).values())

    def _any_spatial_enabled(self) -> bool:
        return any(vars(self.cfg.spatial).values())

    def _any_climatology_enabled(self) -> bool:
        return any(vars(self.cfg.climatology).values())

    def _any_temporal_enabled(self) -> bool:
        return any(vars(self.cfg.temporal).values())

    def _get_metric_arrays(
        self,
        case: Any,
        *,
        family: str,
        metric: str,
        cache: dict[str, tuple[Any, Any]],
    ) -> tuple[Any, Any]:
        product_name = self._resolve_product_name(family=family, metric=metric)
        if product_name in cache:
            return cache[product_name]

        product = self._get_case_product(case, family=family, product_name=product_name)
        forecast = self.loader.get_forecast_array(product)
        target = self.loader.get_target_array(product)
        cache[product_name] = (forecast, target)
        return forecast, target

    def _resolve_product_name(self, *, family: str, metric: str) -> str:
        if family == "probabilistic":
            return get_probabilistic_product(metric)
        if family == "spatial":
            return get_spatial_product(metric)
        if family == "climatology":
            return get_climatology_product(metric)
        if family == "temporal":
            return get_temporal_product(metric)
        raise KeyError(f"Unknown evaluation family {family!r}")

    def _get_case_product(self, case: Any, *, family: str, product_name: str) -> Any:
        """
        Resolve a case product by explicit product name, with a safe fallback to
        the legacy family-based loader methods.
        """
        if hasattr(case, "get_product") and callable(case.get_product):
            try:
                return case.get_product(product_name)
            except Exception:
                pass

        products = getattr(case, "products", None)
        if isinstance(products, dict) and product_name in products:
            return products[product_name]

        if hasattr(case, product_name):
            return getattr(case, product_name)

        logger.debug(
            "Falling back to legacy family-level product routing for family=%s metric=%s product=%s",
            family,
            product_name,
            product_name,
        )

        if family == "probabilistic":
            return self.loader.get_probabilistic_product(case)
        if family == "spatial":
            return self.loader.get_spatial_product(case)
        if family == "climatology":
            return self.loader.get_climatology_product(case)
        if family == "temporal":
            return self.loader.get_temporal_product(case)
        raise KeyError(f"Unknown evaluation family {family!r}")

    def _build_summary_metrics(self) -> None:
        """
        Build compact headline summary metrics from the full evaluation outputs
        and write summary tables.
        """

        summary_rows = summary_metrics_as_dicts(self.metrics)
        self.metrics["summary"] = summary_rows

        if not summary_rows:
            logger.info("No summary metrics produced.")
            return

        try:
            csv_path = write_summary_csv(summary_rows, self.output_dir)
            md_path = write_summary_markdown(summary_rows, self.output_dir)

            logger.info(f"Saved summary CSV to {csv_path}")
            logger.info(f"Saved summary Markdown to {md_path}")

        except Exception as exc:
            logger.warning(f"Failed to write summary tables: {exc}")
