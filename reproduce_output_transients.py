#!/usr/bin/env python3
"""
Reproducibility code for
"Output-specific transient responses in matrix population models:
a scale-consistent framework"

Author
------
Emmanuel Pio Pastore
ORCID: 0009-0007-7851-4414

Suggested repository
--------------------
https://github.com/emmanuel6474/reprocodeforoutputtransients

Purpose
-------
This single script reproduces the synthetic simulation study, summary tables,
and five figures reported in the manuscript. The default settings reproduce the
paper exactly: 750 matrices in each of three demographic regimes, 2,250 matrices
total, seed 20260801, and a principal horizon of 30 projection intervals.

Dependencies
------------
Python 3.10 or later, NumPy, pandas, and Matplotlib.
Install them with:

    python -m pip install numpy pandas matplotlib

Usage
-----
Exact manuscript run:

    python reproduce_output_transients.py

Choose another output directory:

    python reproduce_output_transients.py --output results

Small smoke test, which does not reproduce the manuscript numbers:

    python reproduce_output_transients.py --n-per-regime 10 --no-verify

Outputs
-------
The script creates raw CSV files, summary CSV files, PNG/PDF figures,
run_metadata.txt, and manuscript_verification.txt in the selected output folder.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_N_PER_REGIME = 750
DEFAULT_HORIZON = 30
DEFAULT_SEED = 20260801
REGIMES: Tuple[str, ...] = ("fast", "intermediate", "slow")
RESCALING_EXPONENTS: Tuple[int, ...] = (1, 2, 3, 6)
OUTPUT_CONTRASTS: Tuple[int, ...] = (1, 10, 100, 1000, 1_000_000)
ROBUSTNESS_HORIZONS: Tuple[int, ...] = (5, 10, 20, 30, 50)


# Values printed in the revised manuscript. These checks are intentionally
# limited to load-bearing results and use tolerances that permit harmless
# floating-point variation across supported NumPy versions.
MANUSCRIPT_EXPECTED = {
    "unit_median_ratio_exp1": 1.0573920613802794,
    "unit_median_ratio_exp6": 1.1954957171065588,
    "unit_max_invariance_error": 1.3043767683134166e-12,
    "total_median_peak": 0.2758774424254038,
    "mature_median_peak": 0.46882748052226214,
    "size_median_peak": 0.8491390943199494,
    "total_size_spearman": 0.11767686696497785,
    "total_size_top_decile_overlap": 0.13777777777777778,
    "phase_total_median": 1.2474676476862896,
    "phase_newborn_median": 1.655440495518372,
    "phase_max_spectral_error": 4.605352505700487e-15,
}


Array = np.ndarray
PerronComponents = Tuple[float, Array, Array, Array, Array]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the simulations, summaries, and figures in the manuscript."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproducibility_output"),
        help="Output directory. Default: reproducibility_output",
    )
    parser.add_argument(
        "--n-per-regime",
        type=int,
        default=DEFAULT_N_PER_REGIME,
        help=f"Matrices per regime. Default: {DEFAULT_N_PER_REGIME}",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Principal projection horizon. Default: {DEFAULT_HORIZON}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip comparison with the numerical values reported in the manuscript.",
    )
    args = parser.parse_args()

    if args.n_per_regime < 1:
        parser.error("--n-per-regime must be at least 1")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    return args


def perron(matrix: Array) -> Tuple[float, Array, Array]:
    """Return Perron root, stable-stage vector, and reproductive-value vector.

    The implementation intentionally matches the numerical procedure used for
    the manuscript analyses so that the archived results are reproduced exactly.
    For the non-negative primitive matrices generated here, the dominant right
    and left eigenvectors are unique up to sign.
    """
    eigenvalues, right_eigenvectors = np.linalg.eig(matrix)
    index = int(np.argmax(eigenvalues.real))
    growth_rate = float(eigenvalues[index].real)
    if growth_rate <= 0:
        raise FloatingPointError("The dominant eigenvalue must be positive.")

    stable_stage = np.abs(right_eigenvectors[:, index].real)
    stable_stage /= stable_stage.sum()

    left_eigenvalues, left_eigenvectors = np.linalg.eig(matrix.T)
    left_index = int(np.argmax(left_eigenvalues.real))
    reproductive_value = np.abs(left_eigenvectors[:, left_index].real)
    reproductive_value /= float(reproductive_value @ stable_stage)

    return growth_rate, stable_stage, reproductive_value

def components(matrix: Array) -> PerronComponents:
    growth_rate, stable_stage, reproductive_value = perron(matrix)
    normalized = matrix / growth_rate
    projector = np.outer(stable_stage, reproductive_value)
    return growth_rate, stable_stage, reproductive_value, normalized, projector


def generate_life_history(
    rng: np.random.Generator,
    regime: str,
) -> Tuple[Array, Array, Array]:
    """Generate one sparse primitive stage-structured matrix population model."""
    stage_count = int(rng.integers(3, 8))

    if regime == "fast":
        survival_range = (0.20, 0.70)
        log_fertility_mean = 0.45
    elif regime == "intermediate":
        survival_range = (0.40, 0.86)
        log_fertility_mean = 0.00
    elif regime == "slow":
        survival_range = (0.65, 0.97)
        log_fertility_mean = -0.45
    else:
        raise ValueError(f"Unknown demographic regime: {regime}")

    survival_transition = np.zeros((stage_count, stage_count), dtype=float)

    for stage in range(stage_count):
        survival = float(rng.uniform(*survival_range))
        retrogression_fraction = (
            float(rng.uniform(0.01, 0.12))
            if stage > 0 and rng.random() < 0.25
            else 0.0
        )
        remaining_fraction = 1.0 - retrogression_fraction

        if stage < stage_count - 1:
            stasis_fraction = float(rng.uniform(0.25, 0.75))
            survival_transition[stage, stage] = (
                survival * remaining_fraction * stasis_fraction
            )
            survival_transition[stage + 1, stage] = (
                survival * remaining_fraction * (1.0 - stasis_fraction)
            )
        else:
            survival_transition[stage, stage] = survival * remaining_fraction

        if retrogression_fraction > 0:
            survival_transition[stage - 1, stage] = survival * retrogression_fraction

    fertility = np.zeros((stage_count, stage_count), dtype=float)
    mature_start = max(1, stage_count // 2)
    for stage in range(mature_start, stage_count):
        maturity_modifier = 0.75 + 0.50 * (
            (stage - mature_start) / max(1, stage_count - mature_start - 1)
        )
        fertility[0, stage] = float(
            rng.lognormal(log_fertility_mean, 0.70) * maturity_modifier
        )

    projection = survival_transition + fertility
    return projection, survival_transition, fertility


def observable_series(
    comp: PerronComponents,
    pulse_cost: Array,
    output_weight: Array,
    horizon: int,
) -> Array:
    """Return normalized output deviations for projection times 1 through horizon."""
    _, stable_stage, reproductive_value, normalized, projector = comp

    if np.any(pulse_cost <= 0):
        raise ValueError("Every pulse-cost entry must be strictly positive.")
    if np.any(output_weight < 0) or not np.any(output_weight > 0):
        raise ValueError("Output weights must be non-negative and not all zero.")

    asymptotic_envelope = float(
        (output_weight @ stable_stage) * np.max(reproductive_value / pulse_cost)
    )
    if not np.isfinite(asymptotic_envelope) or asymptotic_envelope <= 0:
        raise FloatingPointError("The asymptotic output envelope was not positive.")

    matrix_power = np.eye(normalized.shape[0])
    deviations: List[float] = []
    for _ in range(horizon):
        matrix_power = matrix_power @ normalized
        residual_output = output_weight @ (matrix_power - projector)
        deviations.append(
            float(np.max(np.abs(residual_output) / pulse_cost) / asymptotic_envelope)
        )
    return np.asarray(deviations)


def observable_metrics(
    comp: PerronComponents,
    pulse_cost: Array,
    output_weight: Array,
    horizon: int,
) -> Dict[str, float]:
    series = observable_series(comp, pulse_cost, output_weight, horizon)
    return {
        "peak": float(series.max()),
        "cumulative": float(series.sum()),
        "peak_time": int(series.argmax() + 1),
    }


def residual_norm_peak(
    comp: PerronComponents,
    order: int,
    horizon: int,
) -> float:
    """Peak normalized full-state residual norm used as the comparator."""
    _, _, _, normalized, projector = comp
    denominator = float(np.linalg.norm(projector, ord=order))
    matrix_power = np.eye(normalized.shape[0])
    values: List[float] = []
    for _ in range(horizon):
        matrix_power = matrix_power @ normalized
        values.append(float(np.linalg.norm(matrix_power - projector, ord=order) / denominator))
    return max(values)


def split_stage_exactly(
    matrix: Array,
    stage: int,
    allocation: float,
) -> Tuple[Array, Array]:
    """Split one stage into two exactly aggregable, interchangeable substages."""
    stage_count = matrix.shape[0]
    aggregation = np.zeros((stage_count, stage_count + 1), dtype=float)
    lifting = np.zeros((stage_count + 1, stage_count), dtype=float)
    refined_index = 0

    for coarse_index in range(stage_count):
        if coarse_index == stage:
            aggregation[coarse_index, refined_index : refined_index + 2] = 1.0
            lifting[refined_index, coarse_index] = allocation
            lifting[refined_index + 1, coarse_index] = 1.0 - allocation
            refined_index += 2
        else:
            aggregation[coarse_index, refined_index] = 1.0
            lifting[refined_index, coarse_index] = 1.0
            refined_index += 1

    refined_matrix = lifting @ matrix @ aggregation
    return refined_matrix, aggregation


def save_boxplot(
    data: Sequence[Iterable[float]],
    labels: Sequence[str],
    ylabel: str,
    title: str,
    stem: str,
    output_dir: Path,
    log_scale: bool = False,
    rotation: float = 0,
) -> None:
    figure, axes = plt.subplots(figsize=(7.2, 4.8))
    axes.boxplot(data, tick_labels=labels, showfliers=False)
    if log_scale:
        axes.set_yscale("log")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.tick_params(axis="x", rotation=rotation)
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.png", dpi=240)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)


def run_simulations(
    output_dir: Path,
    n_per_regime: int,
    horizon: int,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    unit_rows: List[Dict[str, object]] = []
    output_rows: List[Dict[str, object]] = []
    value_flow_rows: List[Dict[str, object]] = []
    phase_rows: List[Dict[str, object]] = []
    horizon_rows: List[Dict[str, object]] = []
    signed_rows: List[Dict[str, object]] = []
    refinement_rows: List[Dict[str, object]] = []
    contrast_rows: List[Dict[str, object]] = []
    input_cost_rows: List[Dict[str, object]] = []
    phase_spectral_errors: List[float] = []

    simulation_id = 0
    robustness_max_horizon = max(ROBUSTNESS_HORIZONS)

    for regime in REGIMES:
        for _ in range(n_per_regime):
            projection, survival_transition, fertility = generate_life_history(rng, regime)
            stage_count = projection.shape[0]
            comp = components(projection)
            _, stable_stage, reproductive_value, normalized, projector = comp

            equal_cost = np.ones(stage_count)
            total_abundance = np.ones(stage_count)
            mature_start = max(1, stage_count // 2)
            mature_abundance = np.concatenate(
                (np.zeros(mature_start), np.ones(stage_count - mature_start))
            )
            size_weighted = np.geomspace(1.0, 100.0, stage_count)

            output_definitions = {
                "Total abundance": total_abundance,
                "Mature-stage abundance": mature_abundance,
                "Size-weighted abundance": size_weighted,
                "Reproductive value": reproductive_value,
            }
            for label, output_weight in output_definitions.items():
                output_rows.append(
                    {
                        "simulation": simulation_id,
                        "regime": regime,
                        "dimension": stage_count,
                        "output": label,
                        **observable_metrics(comp, equal_cost, output_weight, horizon),
                    }
                )

            native_total = observable_metrics(
                comp, equal_cost, total_abundance, horizon
            )
            native_residual_norm = residual_norm_peak(comp, order=1, horizon=horizon)

            for exponent in RESCALING_EXPONENTS:
                factors = 10.0 ** rng.uniform(-exponent, exponent, stage_count)
                diagonal = np.diag(factors)
                transformed_projection = (
                    diagonal @ projection @ np.diag(1.0 / factors)
                )
                transformed_comp = components(transformed_projection)
                transformed_total = observable_metrics(
                    transformed_comp,
                    equal_cost / factors,
                    total_abundance / factors,
                    horizon,
                )
                transformed_residual_norm = residual_norm_peak(
                    transformed_comp, order=1, horizon=horizon
                )
                residual_ratio = transformed_residual_norm / native_residual_norm

                unit_rows.append(
                    {
                        "simulation": simulation_id,
                        "regime": regime,
                        "scale_exponent": exponent,
                        "residual_norm_ratio": residual_ratio,
                        "absolute_log10_norm_change": abs(math.log10(residual_ratio)),
                        "observable_peak_relative_error": abs(
                            transformed_total["peak"] - native_total["peak"]
                        )
                        / max(native_total["peak"], 1e-15),
                        "observable_cumulative_relative_error": abs(
                            transformed_total["cumulative"]
                            - native_total["cumulative"]
                        )
                        / max(native_total["cumulative"], 1e-15),
                    }
                )

            reproductive_value_coordinates = (
                np.diag(reproductive_value)
                @ survival_transition
                @ np.diag(1.0 / reproductive_value)
            )
            stable_stage_coordinates = (
                np.diag(1.0 / stable_stage)
                @ survival_transition
                @ np.diag(stable_stage)
            )
            native_max = float(survival_transition.sum(axis=0).max())
            rv_max = float(reproductive_value_coordinates.sum(axis=0).max())
            stable_max = float(stable_stage_coordinates.sum(axis=0).max())
            value_flow_rows.append(
                {
                    "simulation": simulation_id,
                    "regime": regime,
                    "native_max_column_sum": native_max,
                    "reproductive_value_max_column_sum": rv_max,
                    "stable_stage_max_column_sum": stable_max,
                    "reproductive_value_exceeds_one": float(rv_max > 1.0),
                    "stable_stage_exceeds_one": float(stable_max > 1.0),
                }
            )

            reproductive_pulse = np.eye(stage_count) + fertility
            pre_breeding = survival_transition @ reproductive_pulse
            post_breeding = reproductive_pulse @ survival_transition
            pre_comp = components(pre_breeding)
            post_comp = components(post_breeding)
            phase_spectral_errors.append(
                abs(pre_comp[0] - post_comp[0]) / max(pre_comp[0], 1e-15)
            )

            phase_outputs = {
                "Total abundance": total_abundance,
                "Mature-stage abundance": mature_abundance,
                "Newborn-stage abundance": np.concatenate(
                    (np.array([1.0]), np.zeros(stage_count - 1))
                ),
                "Size-weighted abundance": size_weighted,
            }
            for label, output_weight in phase_outputs.items():
                pre_peak = observable_metrics(
                    pre_comp, equal_cost, output_weight, horizon
                )["peak"]
                post_peak = observable_metrics(
                    post_comp, equal_cost, output_weight, horizon
                )["peak"]
                phase_rows.append(
                    {
                        "simulation": simulation_id,
                        "regime": regime,
                        "output": label,
                        "phase_ratio": max(pre_peak, post_peak)
                        / max(min(pre_peak, post_peak), 1e-15),
                        "pre_peak": pre_peak,
                        "post_peak": post_peak,
                    }
                )

            total_series = observable_series(
                comp, equal_cost, total_abundance, robustness_max_horizon
            )
            for robustness_horizon in ROBUSTNESS_HORIZONS:
                horizon_rows.append(
                    {
                        "simulation": simulation_id,
                        "regime": regime,
                        "horizon": robustness_horizon,
                        "peak": float(total_series[:robustness_horizon].max()),
                        "cumulative": float(total_series[:robustness_horizon].sum()),
                    }
                )

            signed_outputs = {
                "Total abundance": total_abundance,
                "Mature-stage abundance": mature_abundance,
                "Size-weighted abundance": size_weighted,
            }
            for label, output_weight in signed_outputs.items():
                asymptotic_envelope = float(
                    (output_weight @ stable_stage)
                    * np.max(reproductive_value / equal_cost)
                )
                matrix_power = np.eye(stage_count)
                positive: List[float] = []
                negative: List[float] = []
                for _ in range(horizon):
                    matrix_power = matrix_power @ normalized
                    signed_stage_responses = (
                        output_weight @ (matrix_power - projector)
                    ) / equal_cost
                    positive.append(
                        max(0.0, float(signed_stage_responses.max()))
                        / asymptotic_envelope
                    )
                    negative.append(
                        max(0.0, float(-signed_stage_responses.min()))
                        / asymptotic_envelope
                    )
                positive_peak = max(positive)
                negative_peak = max(negative)
                signed_rows.append(
                    {
                        "simulation": simulation_id,
                        "regime": regime,
                        "output": label,
                        "positive_peak": positive_peak,
                        "negative_peak": negative_peak,
                        "dominant_sign": (
                            "amplification"
                            if positive_peak >= negative_peak
                            else "attenuation"
                        ),
                    }
                )

            refined_projection, aggregation = split_stage_exactly(
                projection,
                int(rng.integers(0, stage_count)),
                float(rng.uniform(0.02, 0.98)),
            )
            refined_comp = components(refined_projection)
            refined_total = observable_metrics(
                refined_comp,
                aggregation.T @ equal_cost,
                aggregation.T @ total_abundance,
                horizon,
            )
            refinement_rows.append(
                {
                    "simulation": simulation_id,
                    "regime": regime,
                    "peak_absolute_error": abs(
                        refined_total["peak"] - native_total["peak"]
                    ),
                    "cumulative_absolute_error": abs(
                        refined_total["cumulative"] - native_total["cumulative"]
                    ),
                    "euclidean_residual_ratio": residual_norm_peak(
                        refined_comp, order=2, horizon=horizon
                    )
                    / residual_norm_peak(comp, order=2, horizon=horizon),
                }
            )

            for contrast in OUTPUT_CONTRASTS:
                contrast_weight = np.geomspace(
                    1.0, float(contrast), stage_count
                )
                contrast_rows.append(
                    {
                        "simulation": simulation_id,
                        "regime": regime,
                        "contrast": contrast,
                        **observable_metrics(
                            comp, equal_cost, contrast_weight, horizon
                        ),
                    }
                )

            input_output_scenarios = {
                "Individual pulse to total abundance": (
                    equal_cost,
                    total_abundance,
                ),
                "Individual pulse to size-weighted abundance": (
                    equal_cost,
                    size_weighted,
                ),
                "Size-standardized pulse to size-weighted abundance": (
                    size_weighted,
                    size_weighted,
                ),
                "Reproductive-value-standardized pulse to reproductive value": (
                    reproductive_value,
                    reproductive_value,
                ),
            }
            for label, (pulse_cost, output_weight) in input_output_scenarios.items():
                input_cost_rows.append(
                    {
                        "simulation": simulation_id,
                        "regime": regime,
                        "scenario": label,
                        **observable_metrics(
                            comp, pulse_cost, output_weight, horizon
                        ),
                    }
                )

            simulation_id += 1

    dataframes = {
        "unit_scaling_raw.csv": pd.DataFrame(unit_rows),
        "output_raw.csv": pd.DataFrame(output_rows),
        "value_flow_raw.csv": pd.DataFrame(value_flow_rows),
        "phase_raw.csv": pd.DataFrame(phase_rows),
        "horizon_raw.csv": pd.DataFrame(horizon_rows),
        "signed_raw.csv": pd.DataFrame(signed_rows),
        "refinement_raw.csv": pd.DataFrame(refinement_rows),
        "contrast_raw.csv": pd.DataFrame(contrast_rows),
        "input_cost_raw.csv": pd.DataFrame(input_cost_rows),
    }

    for filename, dataframe in dataframes.items():
        dataframe.to_csv(output_dir / filename, index=False)

    summaries = summarize_results(dataframes, phase_spectral_errors, output_dir)
    make_figures(dataframes, output_dir)
    return {**dataframes, **summaries}


def summarize_results(
    dataframes: Dict[str, pd.DataFrame],
    phase_spectral_errors: Sequence[float],
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    unit = dataframes["unit_scaling_raw.csv"]
    output = dataframes["output_raw.csv"]
    value_flow = dataframes["value_flow_raw.csv"]
    phase = dataframes["phase_raw.csv"]
    horizon = dataframes["horizon_raw.csv"]
    signed = dataframes["signed_raw.csv"]
    refinement = dataframes["refinement_raw.csv"]
    contrast = dataframes["contrast_raw.csv"]
    input_cost = dataframes["input_cost_raw.csv"]

    unit_summary = (
        unit.groupby("scale_exponent")
        .agg(
            median_residual_norm_ratio=("residual_norm_ratio", "median"),
            median_abs_log10_change=("absolute_log10_norm_change", "median"),
            p90_abs_log10_change=(
                "absolute_log10_norm_change",
                lambda values: values.quantile(0.90),
            ),
            max_observable_peak_relative_error=(
                "observable_peak_relative_error",
                "max",
            ),
            max_observable_cumulative_relative_error=(
                "observable_cumulative_relative_error",
                "max",
            ),
        )
        .reset_index()
    )

    output_summary = (
        output.groupby("output")
        .agg(
            median_peak=("peak", "median"),
            p90_peak=("peak", lambda values: values.quantile(0.90)),
            median_cumulative=("cumulative", "median"),
            p90_cumulative=("cumulative", lambda values: values.quantile(0.90)),
            median_peak_time=("peak_time", "median"),
        )
        .reset_index()
    )

    output_regime_summary = (
        output.groupby(["regime", "output"])
        .agg(
            median_peak=("peak", "median"),
            p90_peak=("peak", lambda values: values.quantile(0.90)),
            median_cumulative=("cumulative", "median"),
        )
        .reset_index()
    )

    output_pivot = output.pivot(
        index="simulation", columns="output", values="peak"
    )
    rank_rows: List[Dict[str, object]] = []
    output_pairs = (
        ("Total abundance", "Mature-stage abundance"),
        ("Total abundance", "Size-weighted abundance"),
        ("Mature-stage abundance", "Size-weighted abundance"),
    )
    for first, second in output_pairs:
        first_cutoff = output_pivot[first].quantile(0.90)
        second_cutoff = output_pivot[second].quantile(0.90)
        first_set = set(output_pivot.index[output_pivot[first] >= first_cutoff])
        second_set = set(output_pivot.index[output_pivot[second] >= second_cutoff])
        rank_rows.append(
            {
                "output_a": first,
                "output_b": second,
                "spearman_rho": output_pivot[first].corr(
                    output_pivot[second], method="spearman"
                ),
                "top_decile_overlap_fraction": len(first_set & second_set)
                / len(first_set),
            }
        )
    output_rank_summary = pd.DataFrame(rank_rows)

    value_flow_summary = pd.DataFrame(
        [
            {
                "native_max_column_sum_max": value_flow[
                    "native_max_column_sum"
                ].max(),
                "reproductive_value_proportion_exceeding_one": value_flow[
                    "reproductive_value_exceeds_one"
                ].mean(),
                "stable_stage_proportion_exceeding_one": value_flow[
                    "stable_stage_exceeds_one"
                ].mean(),
            }
        ]
    )

    phase_summary = (
        phase.groupby("output")
        .agg(
            median_phase_ratio=("phase_ratio", "median"),
            p90_phase_ratio=("phase_ratio", lambda values: values.quantile(0.90)),
            p95_phase_ratio=("phase_ratio", lambda values: values.quantile(0.95)),
            proportion_over_1_25=("phase_ratio", lambda values: np.mean(values > 1.25)),
            proportion_over_2=("phase_ratio", lambda values: np.mean(values > 2.0)),
        )
        .reset_index()
    )

    peak_by_horizon = horizon.pivot(
        index="simulation", columns="horizon", values="peak"
    )
    cumulative_by_horizon = horizon.pivot(
        index="simulation", columns="horizon", values="cumulative"
    )
    horizon_rows: List[Dict[str, object]] = []
    reference_horizon = max(ROBUSTNESS_HORIZONS)
    for current_horizon in ROBUSTNESS_HORIZONS[:-1]:
        horizon_rows.extend(
            (
                {
                    "metric": "peak",
                    "horizon": current_horizon,
                    "spearman_with_H50": peak_by_horizon[current_horizon].corr(
                        peak_by_horizon[reference_horizon], method="spearman"
                    ),
                },
                {
                    "metric": "cumulative",
                    "horizon": current_horizon,
                    "spearman_with_H50": cumulative_by_horizon[current_horizon].corr(
                        cumulative_by_horizon[reference_horizon], method="spearman"
                    ),
                },
            )
        )
    horizon_summary = pd.DataFrame(horizon_rows)

    signed_summary = (
        signed.groupby("output")
        .agg(
            proportion_amplification_dominant=(
                "dominant_sign",
                lambda values: np.mean(values == "amplification"),
            ),
            median_positive_peak=("positive_peak", "median"),
            median_negative_peak=("negative_peak", "median"),
        )
        .reset_index()
    )

    refinement_summary = pd.DataFrame(
        [
            {
                "max_peak_absolute_error": refinement["peak_absolute_error"].max(),
                "max_cumulative_absolute_error": refinement[
                    "cumulative_absolute_error"
                ].max(),
                "median_euclidean_residual_ratio": refinement[
                    "euclidean_residual_ratio"
                ].median(),
                "p90_abs_log10_euclidean_change": np.abs(
                    np.log10(refinement["euclidean_residual_ratio"])
                ).quantile(0.90),
                "max_phase_spectral_relative_error": max(phase_spectral_errors),
            }
        ]
    )

    contrast_summary = (
        contrast.groupby("contrast")
        .agg(
            median_peak=("peak", "median"),
            p90_peak=("peak", lambda values: values.quantile(0.90)),
            median_cumulative=("cumulative", "median"),
        )
        .reset_index()
    )

    input_cost_summary = (
        input_cost.groupby("scenario")
        .agg(
            median_peak=("peak", "median"),
            p90_peak=("peak", lambda values: values.quantile(0.90)),
            median_cumulative=("cumulative", "median"),
        )
        .reset_index()
    )

    summaries = {
        "unit_scaling_summary.csv": unit_summary,
        "output_summary.csv": output_summary,
        "output_regime_summary.csv": output_regime_summary,
        "output_rank_summary.csv": output_rank_summary,
        "value_flow_summary.csv": value_flow_summary,
        "phase_summary.csv": phase_summary,
        "horizon_summary.csv": horizon_summary,
        "signed_summary.csv": signed_summary,
        "refinement_summary.csv": refinement_summary,
        "contrast_summary.csv": contrast_summary,
        "input_cost_summary.csv": input_cost_summary,
    }
    for filename, dataframe in summaries.items():
        dataframe.to_csv(output_dir / filename, index=False)
    return summaries


def make_figures(dataframes: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    unit = dataframes["unit_scaling_raw.csv"]
    output = dataframes["output_raw.csv"]
    value_flow = dataframes["value_flow_raw.csv"]
    phase = dataframes["phase_raw.csv"]
    input_cost = dataframes["input_cost_raw.csv"]

    save_boxplot(
        [
            unit.loc[
                unit["scale_exponent"] == exponent,
                "absolute_log10_norm_change",
            ]
            for exponent in RESCALING_EXPONENTS
        ],
        [
            rf"$10^{{-{exponent}}}$ to $10^{{{exponent}}}$"
            for exponent in RESCALING_EXPONENTS
        ],
        "Absolute change in log10 residual-norm peak",
        "Unweighted full-state responses depend on stage units",
        "fig1_unit_dependence",
        output_dir,
    )

    output_labels = (
        "Total abundance",
        "Mature-stage abundance",
        "Size-weighted abundance",
    )
    save_boxplot(
        [
            output.loc[output["output"] == label, "peak"]
            for label in output_labels
        ],
        output_labels,
        "Peak normalized output deviation",
        "Transient response depends on the measured output",
        "fig2_output_dependence",
        output_dir,
        log_scale=True,
        rotation=12,
    )

    save_boxplot(
        [
            value_flow["native_max_column_sum"],
            value_flow["reproductive_value_max_column_sum"],
            value_flow["stable_stage_max_column_sum"],
        ],
        ("Native", "Reproductive-value", "Stable-stage"),
        "Maximum column sum of survival-transition component",
        "Transformed process matrices describe value flow",
        "fig3_value_flow",
        output_dir,
        log_scale=True,
    )

    phase_labels = (
        "Total abundance",
        "Mature-stage abundance",
        "Newborn-stage abundance",
        "Size-weighted abundance",
    )
    save_boxplot(
        [phase.loc[phase["output"] == label, "phase_ratio"] for label in phase_labels],
        phase_labels,
        "Larger-to-smaller phase-specific peak ratio",
        "Census phase changes output-specific transient responses",
        "fig4_phase_dependence",
        output_dir,
        log_scale=True,
        rotation=15,
    )

    scenario_order = (
        "Individual pulse to total abundance",
        "Individual pulse to size-weighted abundance",
        "Size-standardized pulse to size-weighted abundance",
    )
    save_boxplot(
        [
            input_cost.loc[input_cost["scenario"] == scenario, "peak"]
            for scenario in scenario_order
        ],
        ("Individual to total", "Individual to size-weighted", "Size to size-weighted"),
        "Peak normalized output deviation",
        "Pulse cost and output weighting define different interventions",
        "fig5_input_output",
        output_dir,
        log_scale=True,
        rotation=10,
    )


def extract_verification_values(results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    unit = results["unit_scaling_summary.csv"].set_index("scale_exponent")
    output = results["output_summary.csv"].set_index("output")
    rank = results["output_rank_summary.csv"]
    phase = results["phase_summary.csv"].set_index("output")
    refinement = results["refinement_summary.csv"].iloc[0]

    total_size = rank[
        (rank["output_a"] == "Total abundance")
        & (rank["output_b"] == "Size-weighted abundance")
    ].iloc[0]

    return {
        "unit_median_ratio_exp1": float(unit.loc[1, "median_residual_norm_ratio"]),
        "unit_median_ratio_exp6": float(unit.loc[6, "median_residual_norm_ratio"]),
        "unit_max_invariance_error": float(
            unit["max_observable_peak_relative_error"].max()
        ),
        "total_median_peak": float(output.loc["Total abundance", "median_peak"]),
        "mature_median_peak": float(
            output.loc["Mature-stage abundance", "median_peak"]
        ),
        "size_median_peak": float(
            output.loc["Size-weighted abundance", "median_peak"]
        ),
        "total_size_spearman": float(total_size["spearman_rho"]),
        "total_size_top_decile_overlap": float(
            total_size["top_decile_overlap_fraction"]
        ),
        "phase_total_median": float(
            phase.loc["Total abundance", "median_phase_ratio"]
        ),
        "phase_newborn_median": float(
            phase.loc["Newborn-stage abundance", "median_phase_ratio"]
        ),
        "phase_max_spectral_error": float(
            refinement["max_phase_spectral_relative_error"]
        ),
    }


def verify_manuscript_values(
    results: Dict[str, pd.DataFrame],
    output_dir: Path,
    exact_default_run: bool,
) -> bool:
    report_path = output_dir / "manuscript_verification.txt"
    if not exact_default_run:
        report_path.write_text(
            "Verification skipped because the run did not use the manuscript defaults.\n",
            encoding="utf-8",
        )
        return True

    actual = extract_verification_values(results)
    lines = ["MANUSCRIPT NUMERICAL VERIFICATION", ""]
    all_passed = True

    for key, expected in MANUSCRIPT_EXPECTED.items():
        observed = actual[key]
        if abs(expected) < 1e-10:
            tolerance = 2e-11
        else:
            tolerance = 5e-10 + 5e-8 * abs(expected)
        passed = abs(observed - expected) <= tolerance
        all_passed = all_passed and passed
        lines.append(
            f"{'PASS' if passed else 'FAIL'}  {key}: "
            f"observed={observed:.16g}, expected={expected:.16g}, "
            f"absolute_difference={abs(observed - expected):.3g}"
        )

    lines.extend(("", f"OVERALL: {'PASS' if all_passed else 'FAIL'}"))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_passed


def write_metadata(
    output_dir: Path,
    n_per_regime: int,
    horizon: int,
    seed: int,
) -> None:
    metadata = [
        "Output-specific transient responses: reproducibility run",
        f"Python: {sys.version.replace(chr(10), ' ')}",
        f"Platform: {platform.platform()}",
        f"NumPy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"Matplotlib: {matplotlib.__version__}",
        f"Seed: {seed}",
        f"Matrices per regime: {n_per_regime}",
        f"Regimes: {', '.join(REGIMES)}",
        f"Total matrices: {n_per_regime * len(REGIMES)}",
        f"Principal horizon: {horizon}",
    ]
    (output_dir / "run_metadata.txt").write_text(
        "\n".join(metadata) + "\n", encoding="utf-8"
    )


def print_key_results(results: Dict[str, pd.DataFrame]) -> None:
    print("\nUNIT SCALING")
    print(results["unit_scaling_summary.csv"].to_string(index=False))
    print("\nOUTPUTS")
    print(results["output_summary.csv"].to_string(index=False))
    print("\nOUTPUT RANKS")
    print(results["output_rank_summary.csv"].to_string(index=False))
    print("\nVALUE-FLOW COORDINATES")
    print(results["value_flow_summary.csv"].to_string(index=False))
    print("\nCENSUS PHASE")
    print(results["phase_summary.csv"].to_string(index=False))
    print("\nSIGNED RESPONSES")
    print(results["signed_summary.csv"].to_string(index=False))
    print("\nREFINEMENT AND SPECTRAL CHECKS")
    print(results["refinement_summary.csv"].to_string(index=False))


def main() -> int:
    args = parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_default_run = (
        args.n_per_regime == DEFAULT_N_PER_REGIME
        and args.horizon == DEFAULT_HORIZON
        and args.seed == DEFAULT_SEED
    )

    print(
        f"Running {args.n_per_regime * len(REGIMES)} simulations "
        f"with seed {args.seed}."
    )
    results = run_simulations(
        output_dir=output_dir,
        n_per_regime=args.n_per_regime,
        horizon=args.horizon,
        seed=args.seed,
    )
    write_metadata(output_dir, args.n_per_regime, args.horizon, args.seed)

    verification_passed = True
    if not args.no_verify:
        verification_passed = verify_manuscript_values(
            results, output_dir, exact_default_run
        )

    print_key_results(results)
    print(f"\nOutputs written to: {output_dir}")
    if not args.no_verify:
        print(
            "Manuscript verification: "
            + ("PASS" if verification_passed else "FAIL")
        )

    return 0 if verification_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
