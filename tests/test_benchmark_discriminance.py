"""
tests/test_benchmark_discriminance.py
=====================================
Measures the **discriminative power** of the Lab6 benchmark.

The question these tests answer is not "does the model reproduce TerraME?" but
"can the benchmark tell a correct CLUE-S implementation from one that does not
implement CLUE-S at all?".

Reproducing the reference is necessary but not sufficient. If a trivial baseline
reaches the same parity, the benchmark is not measuring the algorithm.

Known state (2026-07-27)
------------------------
The naive baseline — a static ranking by ``prob_d - prob_f``, with no CLUE-S, no
iteration and no time steps — reproduces the TerraME output **exactly**, cell for
cell, 5914/5914. The Lab6 scenario therefore validates only the transcription of
the logistic regression coefficients, **not** the allocation algorithm.

That is why ``test_benchmark_is_discriminative`` is marked as a strict xfail.
Once the scenario is made discriminative (see that test's docstring), it will
XPASS and the marker must be removed.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
BENCHMARK_DIR = pathlib.Path(__file__).parent.parent / "benchmark" / "data"
INPUT_ZIP = DATA_DIR / "cs_moju.zip"
TERRAME_ZIP = BENCHMARK_DIR / "Lab6_2004.zip"

skip_if_no_data = pytest.mark.skipif(
    not INPUT_ZIP.exists() or not TERRAME_ZIP.exists(),
    reason="Lab6 data files not found",
)


def _load():
    import geopandas as gpd

    return gpd.read_file(str(INPUT_ZIP)), gpd.read_file(str(TERRAME_ZIP))


def _reference(gdf_terrame) -> np.ndarray:
    col = "d_out" if "d_out" in gdf_terrame.columns else "d"
    return (gdf_terrame[col].values >= 0.5).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Characterisation — does the naive baseline reproduce the reference?
# ══════════════════════════════════════════════════════════════════════════════

@skip_if_no_data
def test_naive_baseline_reproduces_terrame():
    """Records that a trivial static ranking already reproduces TerraME.

    This is NOT a requirement on the model — it characterises a limitation of
    the benchmark scenario. If it ever fails, the scenario has stopped being
    trivially reducible, which would be an improvement.
    """
    from benchmark.naive_baseline import naive_allocation

    gdf_input, gdf_terrame = _load()
    ref = _reference(gdf_terrame)

    pred = naive_allocation(gdf_input, n_target=int(ref.sum()))
    agreement = float((pred == ref).mean()) * 100

    assert agreement == pytest.approx(100.0, abs=1e-9), (
        f"The naive baseline agrees with TerraME on {agreement:.4f}% of cells. "
        "If this is no longer 100%, the scenario has changed — revisit "
        "test_benchmark_is_discriminative, which may have stopped being xfail."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Discriminance — does the benchmark tell CLUE-S from non-CLUE-S?
# ══════════════════════════════════════════════════════════════════════════════

@skip_if_no_data
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limitation: in the Lab6 scenario every covariate is static, the "
        "elasticity of f is 0.0 and d is irreversible, so allocation collapses "
        "into a static threshold. The naive baseline ties with CLUE-S. Making "
        "the benchmark discriminative requires a scenario with a dynamic "
        "covariate (for example, distance to deforestation updated every step), a "
        "reversible d->f transition, or multiple regions."
    ),
)
def test_benchmark_is_discriminative():
    """Full CLUE-S should outperform the naive baseline.

    While the two tie, the benchmark does not constrain the allocation
    algorithm — it merely checks that the regression coefficients were
    transcribed correctly.
    """
    from benchmark.naive_baseline import naive_allocation
    from disslucc_discrete.executors.lucc_validation_executor import (
        _align_on_col_lin,
        _run_python,
    )

    gdf_input, gdf_terrame = _load()
    ref_full = _reference(gdf_terrame)

    gdf_result, _ = _run_python(gdf_input.copy())
    df = _align_on_col_lin(gdf_result, gdf_terrame)
    agreement_clue = float(
        ((df["d_py"].values >= 0.5).astype(int)
         == (df["d_terrame"].values >= 0.5).astype(int)).mean()
    ) * 100

    pred_naive = naive_allocation(gdf_input, n_target=int(ref_full.sum()))
    agreement_naive = float((pred_naive == ref_full).mean()) * 100

    assert agreement_clue > agreement_naive, (
        f"CLUE-S={agreement_clue:.4f}% vs naive baseline={agreement_naive:.4f}%. "
        "The benchmark cannot tell the algorithm from a trivial static ranking."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sensitivity — does the demand trajectory influence the result?
# ══════════════════════════════════════════════════════════════════════════════

@skip_if_no_data
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limitation: because d is irreversible and demand is monotonic, "
        "the final state depends only on the final demand, not on the path. "
        "Changing the intermediate trajectory does not change the result."
    ),
)
def test_trajectory_affects_result():
    """A different demand trajectory should produce a different result.

    If the final state is identical whether running 6 steps or 2 steps with an
    invented intermediate demand, the benchmark validates no temporal dynamics
    at all.
    """
    import geopandas as gpd
    from dissmodel.core import Environment
    from disslucc_discrete.components.allocation.vector.clue_s import (
        AllocationDClueSLike,
    )
    from disslucc_discrete.components.demand.precomputed import (
        DemandPreComputedValues,
    )
    from disslucc_discrete.components.potential.vector.logistic_regression import (
        PotentialDLogisticRegression,
    )
    from disslucc_discrete.executors.lucc_validation_executor import (
        ANNUAL_DEMAND,
        LAND_USE_TYPES,
        POTENTIAL_DATA,
        TRANSITION_MATRIX,
    )

    def run(demand):
        gdf = gpd.read_file(str(INPUT_ZIP))
        env = Environment(end_time=len(demand) - 1)
        dem = DemandPreComputedValues(
            annual_demand=demand, land_use_types=LAND_USE_TYPES
        )
        PotentialDLogisticRegression(
            gdf=gdf, potential_data=POTENTIAL_DATA, land_use_types=LAND_USE_TYPES
        )
        AllocationDClueSLike(
            gdf=gdf, demand=dem, land_use_types=LAND_USE_TYPES,
            transition_matrix=TRANSITION_MATRIX, cell_area=1.0,
            max_difference=10.0, max_iteration=5000, factor_iteration=0.0001,
        )
        env.run()
        return (gdf["d"].values >= 0.5).astype(int)

    official = run(ANNUAL_DEMAND[:6])

    # Same final demand, arbitrary intermediate path
    first, last = ANNUAL_DEMAND[0], ANNUAL_DEMAND[5]
    midpoint = [(a + b) / 2 for a, b in zip(first, last)]
    alternative = run([first, midpoint, last])

    differing = int((official != alternative).sum())
    assert differing > 0, (
        "The intermediate trajectory does not affect the final result "
        f"({differing} differing cells). The benchmark validates no temporal "
        "dynamics."
    )
