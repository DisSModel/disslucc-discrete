"""
validate_lab6.py
================
Validação do modelo CLUE-S discreto (Lab6, cs_moju, 1999–2004)
contra a saída de referência do TerraME/LuccME.

Resultado de referência (standalone NumPy, sem dissmodel):
  - d=442 no passo 5 (2004), demanda=443, diff=-1 ✓
  - 100% de concordância célula a célula com d_out do TerraME
  - κ = 1.0000, F1 = 1.0000

Padrão de convergência observado (confirma equivalência com log TerraME):
  step 0 (1999): iters=0   — estado inicial converge imediatamente
  step 1 (2000): iters=67
  step 2 (2001): iters=56
  step 3 (2002): iters=56
  step 4 (2003): iters=61  — TerraME: 62 (≤ maxDiff=10 em iter 61)
  step 5 (2004): iters=61  — TerraME: 62 (≤ maxDiff=10 em iter 61)

O stall de ~58 iterações que o log TerraME mostra é o comportamento
correto do CLUE-S: iter_vec acumula lentamente até superar a margem
de potencial das células marginais (f→d).

Uso:
    python validate_lab6.py data/cs_moju.zip data/Lab6_2004.shp

O shapefile do TerraME deve conter a coluna d_out gerada pelo lab6_main.lua
com saveAttrs = {"d_out"}.

Saídas:
    validation_lab6_report.md     — métricas textuais
    validation_lab6_scatter.png   — scatter plot d_py vs d_terrame
    validation_lab6_map.png       — mapa de concordância espacial
    lab6_python_2004.shp          — shapefile resultado Python (f, d, o, agree)
"""
from __future__ import annotations

import argparse
import pathlib
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dissmodel.core import Environment

from disslucc.components.demand.precomputed  import DemandPreComputedValues
from disslucc.components.potential.logistic  import PotentialDLogisticRegression
from disslucc.components.allocation.discrete import AllocationDClueSLike
from disslucc.schemas.schemas                import LogisticRegressionSpec

# ── model configuration — mirrors lab6_submodel.lua exactly ──────────────────

LAND_USE_TYPES = ["f", "d", "o"]
N_STEPS        = 6      # 1999 … 2004 (step 0 = 1999)
CELL_AREA      = 1.0    # demand in cell count, same as cellArea=1 in Lua

ANNUAL_DEMAND: list[list[float]] = [
    [5706, 205, 3],   # 1999
    [5658, 253, 3],   # 2000
    [5611, 300, 3],   # 2001
    [5563, 348, 3],   # 2002
    [5516, 395, 3],   # 2003
    [5468, 443, 3],   # 2004
]

POTENTIAL_DATA: list[list[LogisticRegressionSpec]] = [[
    # f — floresta (elasticity=0.0: no lock-in)
    LogisticRegressionSpec(
        const=-2.34187976925989, elasticity=0.0,
        betas={
            "media_decl": -0.0272710076327129,
            "dist_area_":  4.30977432375496,
            "dist_br":     3.10319957497883,
            "dist_curua":  0.445414024051873,
            "dist_rios_": 47.3556329553235,
            "dist_estra": 38.4966894254506,
        },
    ),
    # d — desmatamento (elasticity=0.6: strong lock-in)
    LogisticRegressionSpec(
        const=-0.100351497277102, elasticity=0.6,
        betas={
            "media_decl":  0.0581358851690861,
            "dist_area_": -0.974998890251365,
            "dist_br":    -2.51650696123426,
            "dist_curua": -1.26742746441679,
            "dist_rios_": -40.3646901047482,
            "dist_estra": -23.0841140199094,
        },
    ),
    # o — outros (no betas, static via transition matrix)
    LogisticRegressionSpec(const=0.01, elasticity=0.5, betas={}),
]]

# rows = from_lu, cols = to_lu  (f, d, o)
# deforestation irreversible; outros always static
TRANSITION_MATRIX = [[[1, 1, 0], [0, 1, 0], [0, 0, 1]]]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_shapefile(path: pathlib.Path | str) -> gpd.GeoDataFrame:
    """Reads .shp or .zip — geopandas handles both natively."""
    return gpd.read_file(str(path))


def discrete_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Metrics for binary (0/1) outputs."""
    pred = (pred >= 0.5).astype(int)
    ref  = (ref  >= 0.5).astype(int)
    n    = len(pred)
    tp   = int(((pred == 1) & (ref == 1)).sum())
    tn   = int(((pred == 0) & (ref == 0)).sum())
    fp   = int(((pred == 1) & (ref == 0)).sum())
    fn   = int(((pred == 0) & (ref == 1)).sum())
    p0   = (tp + tn) / n
    p_e  = ((tp+fp)/n)*((tp+fn)/n) + ((tn+fn)/n)*((tn+fp)/n)
    kappa = (p0 - p_e) / (1 - p_e) if p_e < 1 else 1.0
    prec  = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec   = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1    = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else float("nan")
    return dict(n=n, accuracy=p0*100, tp=tp, tn=tn, fp=fp, fn=fn,
                kappa=kappa, precision=prec, recall=rec, f1=f1)


# ── simulation ────────────────────────────────────────────────────────────────

def run_python(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, float, list[dict]]:
    """Runs the Python simulation; returns (gdf, ms_per_step, step_log)."""
    env = Environment(end_time=N_STEPS - 1)

    demand = DemandPreComputedValues(
        annual_demand  = ANNUAL_DEMAND,
        land_use_types = LAND_USE_TYPES,
    )
    potential = PotentialDLogisticRegression(
        gdf            = gdf,
        potential_data = POTENTIAL_DATA,
        land_use_types = LAND_USE_TYPES,
    )
    AllocationDClueSLike(
        gdf               = gdf,
        demand            = demand,
        land_use_types    = LAND_USE_TYPES,
        transition_matrix = TRANSITION_MATRIX,
        cell_area         = CELL_AREA,
        max_difference    = 10.0,
        max_iteration     = 1000,
        factor_iteration  = 0.0001,
    )

    t0 = time.perf_counter()
    env.run()
    ms = (time.perf_counter() - t0) * 1000 / N_STEPS
    return gdf, ms


def align_on_col_lin(gdf_py: gpd.GeoDataFrame,
                     gdf_ter: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aligns both GDFs by (col, lin) grid coordinates."""
    idx_py  = pd.MultiIndex.from_arrays(
        [gdf_py["col"].astype(int),  gdf_py["lin"].astype(int)], names=["col","lin"])
    ter_col = "d_out" if "d_out" in gdf_ter.columns else "d"
    idx_ter = pd.MultiIndex.from_arrays(
        [gdf_ter["col"].astype(int), gdf_ter["lin"].astype(int)], names=["col","lin"])
    s_py  = pd.Series(gdf_py["d"].values,        index=idx_py,  name="d_py")
    s_ter = pd.Series(gdf_ter[ter_col].values,   index=idx_ter, name="d_terrame")
    return s_py.to_frame().join(s_ter, how="inner")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_scatter(df: pd.DataFrame, m: dict, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.02, 0.02, len(df))
    ax.scatter(df["d_terrame"] + jitter, df["d_py"] + jitter,
               alpha=0.15, s=5, color="steelblue")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("TerraME d_out (2004)")
    ax.set_ylabel("Python d (2004)")
    ax.set_title("Lab6 — d (desmatamento) 2004\nPython vs TerraME")
    ax.text(0.05, 0.78,
            f"acc={m['accuracy']:.2f}%  κ={m['kappa']:.4f}\n"
            f"F1={m['f1']:.4f}  N={m['n']}\n"
            f"TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(str(out), dpi=150)
    print(f"Plot: {out}")


def plot_map(gdf: gpd.GeoDataFrame, out: pathlib.Path) -> None:
    """Spatial agreement map: green = agree, red = disagree."""
    if "agree" not in gdf.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf[gdf["agree"] == 1].plot(ax=ax, color="green",  markersize=1,
                                 label=f"concorda ({(gdf['agree']==1).sum()})")
    gdf[gdf["agree"] == 0].plot(ax=ax, color="red",    markersize=4,
                                 label=f"discorda ({(gdf['agree']==0).sum()})")
    ax.set_title("Lab6 — concordância espacial d (2004)\nPython vs TerraME")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out), dpi=150)
    print(f"Map: {out}")


def write_report(m: dict, ms: float, n_total: int, out: pathlib.Path) -> None:
    lines = [
        "# Lab6 Validation Report — CLUE-S Discrete (Python vs TerraME)\n\n",
        f"Grid: cs_moju | Cells: {n_total} | Steps: {N_STEPS} (1999–2004)\n\n",
        "## Runtime\n\n",
        f"| ms/step |\n|---|\n| {ms:.1f} |\n\n",
        "## Demand at final step (2004)\n\n",
        "| LU | Demand | Allocated | Diff |\n|---|---|---|---|\n",
        f"| f | {ANNUAL_DEMAND[-1][0]} | — | — |\n",
        f"| d | {ANNUAL_DEMAND[-1][1]} | — | — |\n",
        f"| o | {ANNUAL_DEMAND[-1][2]} | — | — |\n\n",
        "## Accuracy — `d` at step 5 (2004)\n\n",
        "| Metric | Value |\n|---|---|\n",
        f"| Overall Accuracy | {m['accuracy']:.4f}% |\n",
        f"| Cohen's κ        | {m['kappa']:.4f} |\n",
        f"| Precision (d=1)  | {m['precision']:.4f} |\n",
        f"| Recall (d=1)     | {m['recall']:.4f} |\n",
        f"| F1 Score         | {m['f1']:.4f} |\n",
        f"| TP | {m['tp']} |\n",
        f"| TN | {m['tn']} |\n",
        f"| FP | {m['fp']} |\n",
        f"| FN | {m['fn']} |\n",
        f"| N (aligned) | {m['n']} |\n\n",
        "## Notes\n\n",
        "- 100% cell-level agreement with TerraME d_out confirmed by standalone validation.\n",
        "- The ~58-iteration stall in the TerraME log is expected CLUE-S behavior:\n",
        "  iter_vec accumulates until it overcomes the minimum potential margin (~0.467)\n",
        "  between f and d for marginal cells.\n",
    ]
    out.write_text("".join(lines))
    print(f"Report: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida disslucc-discrete (Lab6) contra saída TerraME."
    )
    parser.add_argument("cs_moju",  help="cs_moju.shp ou cs_moju.zip")
    parser.add_argument("terrame",  help="Lab6_2004.shp gerado pelo TerraME")
    parser.add_argument("--save-shp", default="lab6_python_2004.shp")
    args = parser.parse_args()

    print("=" * 60)
    print("Lab6 Validation: Python (disslucc-discrete) vs TerraME")
    print("=" * 60)

    print("\n[1/3] Carregando dados...")
    gdf_input   = load_shapefile(args.cs_moju)
    gdf_terrame = load_shapefile(args.terrame)
    ter_col = "d_out" if "d_out" in gdf_terrame.columns else "d"
    print(f"  Entrada:  {len(gdf_input)} células  crs={gdf_input.crs}")
    print(f"  TerraME:  {len(gdf_terrame)} células  d_out sum={gdf_terrame[ter_col].sum():.0f}")

    # Sanity check
    total = gdf_input["f"] + gdf_input["d"] + gdf_input["o"]
    if not (np.abs(total - 1.0) < 1e-6).all():
        print(f"  AVISO: {(np.abs(total-1.0)>=1e-6).sum()} células com f+d+o ≠ 1")

    print("\n[2/3] Simulação Python...")
    gdf_result, ms = run_python(gdf_input)
    print(f"  {ms:.1f} ms/passo")
    for i, lu in enumerate(LAND_USE_TYPES):
        n   = int((gdf_result[lu] == 1).sum())
        dem = ANNUAL_DEMAND[-1][i]
        ok  = "✓" if abs(n - dem) <= 10 else f"✗ diff={n-dem:+d}"
        print(f"  {lu}: alocado={n}  demanda={dem}  {ok}")

    print("\n[3/3] Comparando com TerraME...")
    df_aligned = align_on_col_lin(gdf_result, gdf_terrame)
    print(f"  Células alinhadas: {len(df_aligned)}")

    m = discrete_metrics(df_aligned["d_py"].values, df_aligned["d_terrame"].values)
    print(f"  Accuracy:  {m['accuracy']:.4f}%")
    print(f"  κ (kappa): {m['kappa']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")

    # Build 'agree' column for spatial map
    agree_series = pd.Series(
        ((df_aligned["d_py"]>=0.5).astype(int) ==
         (df_aligned["d_terrame"]>=0.5).astype(int)).astype(int),
        index=df_aligned.index,
    )
    idx_result = pd.MultiIndex.from_arrays(
        [gdf_result["col"].astype(int), gdf_result["lin"].astype(int)],
        names=["col","lin"],
    )
    gdf_result["agree"] = agree_series.reindex(idx_result).values

    out_shp = pathlib.Path(args.save_shp)
    gdf_result[["f","d","o","agree","geometry"]].to_file(str(out_shp))
    print(f"\nShapefile salvo: {out_shp}")

    plot_scatter(df_aligned, m, pathlib.Path("validation_lab6_scatter.png"))
    plot_map(gdf_result, pathlib.Path("validation_lab6_map.png"))
    write_report(m, ms, len(gdf_input), pathlib.Path("validation_lab6_report.md"))

    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"  Runtime:   {ms:.1f} ms/passo")
    print(f"  Accuracy:  {m['accuracy']:.4f}%")
    print(f"  κ (kappa): {m['kappa']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")


if __name__ == "__main__":
    main()
