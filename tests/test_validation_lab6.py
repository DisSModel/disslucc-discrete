"""
tests/test_validation_lab6.py
------------------------------
Integration test: verifies 100% cell-level parity between the Python CLUE-S
implementation and the TerraME/LuccME reference (Lab6, cs_moju, 1999–2004).

Expected results (confirmed against TerraME log):
  accuracy = 100.0%  kappa = 1.0  f1 = 1.0
"""
import pathlib
import pytest

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
BENCHMARK_DIR = pathlib.Path(__file__).parent.parent / "benchmark" / "data"
INPUT_ZIP = DATA_DIR / "cs_moju.zip"
TERRAME_ZIP = BENCHMARK_DIR / "Lab6_2004.zip"


@pytest.mark.skipif(
    not INPUT_ZIP.exists() or not TERRAME_ZIP.exists(),
    reason="Lab6 data files not found — skipping integration test",
)
def test_lab6_parity():
    """Python CLUE-S must achieve 100% cell-level agreement with TerraME."""
    import geopandas as gpd
    from disslucc_discrete.executors.lucc_validation_executor import (
        ANNUAL_DEMAND,
        LAND_USE_TYPES,
        _align_on_col_lin,
        _discrete_metrics,
        _run_python,
    )

    gdf_input = gpd.read_file(str(INPUT_ZIP))
    gdf_terrame = gpd.read_file(str(TERRAME_ZIP))

    gdf_result, _ = _run_python(gdf_input)
    df_aligned = _align_on_col_lin(gdf_result, gdf_terrame)
    m = _discrete_metrics(df_aligned["d_py"].values, df_aligned["d_terrame"].values)

    assert m["accuracy"] == pytest.approx(100.0, abs=1e-4), (
        f"Expected accuracy=100%, got {m['accuracy']:.4f}%"
    )
    # Pontius & Millones (2011) — replaces kappa as the parity criterion.
    assert m["quantity_disagreement"] == pytest.approx(0.0, abs=1e-9), (
        f"Quantity disagreement={m['quantity_disagreement']:.6f}, expected 0"
    )
    assert m["allocation_disagreement"] == pytest.approx(0.0, abs=1e-9), (
        f"Allocation disagreement={m['allocation_disagreement']:.6f}, expected 0"
    )
    # Identidade de Pontius: quantity + allocation == 1 - accuracy
    assert m["total_disagreement"] == pytest.approx(
        1.0 - m["accuracy"] / 100, abs=1e-9
    ), "Pontius identity violated — check _discrete_metrics"
    assert m["f1"] == pytest.approx(1.0, abs=1e-4), (
        f"Expected F1=1.0, got {m['f1']:.4f}"
    )
    assert m["fp"] == 0, f"Expected FP=0, got {m['fp']}"
    assert m["fn"] == 0, f"Expected FN=0, got {m['fn']}"

    # Final demand must be met within tolerance
    n_d = int((gdf_result["d"] == 1).sum())
    expected_d = int(ANNUAL_DEMAND[-1][LAND_USE_TYPES.index("d")])
    assert abs(n_d - expected_d) <= 10, (
        f"Final d allocation={n_d}, demand={expected_d}, diff={n_d - expected_d:+d}"
    )
