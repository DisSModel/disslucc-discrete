"""
validate_lab6.py
================
Validação do modelo CLUE-S discreto (Lab6, cs_moju, 1999-2004)
contra a saída de referência do TerraME/LuccME.

Uso:
    python validate_lab6.py data/cs_moju.zip data/terrame_lab6_2004.shp

O shapefile do TerraME deve conter a coluna `d_out` (ou `d`) para 2004,
conforme gerado por lab6_main.lua com saveAttrs = {"d_out"}.

Saídas:
    validation_lab6_report.md     — métricas textuais
    validation_lab6_scatter.png   — scatter plot d_py vs d_terrame
    validation_lab6_map.png       — mapa de concordância espacial
    lab6_python_2004.shp          — shapefile resultado do Python (f, d, o + agree)
"""
from __future__ import annotations

import argparse
import pathlib
import time
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dissmodel.core import Environment

from disslucc.components.demand.precomputed      import DemandPreComputedValues
from disslucc.components.potential.logistic      import PotentialDLogisticRegression
from disslucc.components.allocation.discrete     import AllocationDClueSLike
from disslucc.schemas.schemas                    import LogisticRegressionSpec

# ── model configuration (mirrors lab6_submodel.lua exactly) ──────────────────

LAND_USE_TYPES = ["f", "d", "o"]
N_STEPS        = 6    # 1999 … 2004  (step 0 = 1999, step 5 = 2004)
CELL_AREA      = 1.0  # demanda em número de células, como no Lua

ANNUAL_DEMAND: list[list[float]] = [
    [5706, 205, 3],   # 1999  step 0
    [5658, 253, 3],   # 2000  step 1
    [5611, 300, 3],   # 2001  step 2
    [5563, 348, 3],   # 2002  step 3
    [5516, 395, 3],   # 2003  step 4
    [5468, 443, 3],   # 2004  step 5
]

# Region 1 — logistic regression parameters from lab6_submodel.lua
POTENTIAL_DATA: list[list[LogisticRegressionSpec]] = [[
    # f — floresta (elasticity = 0.0: sem autorreforço)
    LogisticRegressionSpec(
        const      = -2.34187976925989,
        elasticity = 0.0,
        betas      = {
            "media_decl": -0.0272710076327129,
            "dist_area_":  4.30977432375496,
            "dist_br":     3.10319957497883,
            "dist_curua":  0.445414024051873,
            "dist_rios_": 47.3556329553235,
            "dist_estra": 38.4966894254506,
        },
    ),
    # d — desmatamento (elasticity = 0.6: forte lock-in)
    LogisticRegressionSpec(
        const      = -0.100351497277102,
        elasticity = 0.6,
        betas      = {
            "media_decl":  0.0581358851690861,
            "dist_area_": -0.974998890251365,
            "dist_br":    -2.51650696123426,
            "dist_curua": -1.26742746441679,
            "dist_rios_": -40.3646901047482,
            "dist_estra": -23.0841140199094,
        },
    ),
    # o — outros (sem betas, elasticity = 0.5, sempre estático via transition_matrix)
    LogisticRegressionSpec(
        const      = 0.01,
        elasticity = 0.5,
        betas      = {},
    ),
]]

# Transition matrix (Region 1) — mirrors lab6_submodel.lua exactly:
#   f → f ✓  f → d ✓  f → o ✗
#   d → f ✗  d → d ✓  d → o ✗   (desmatamento irreversível)
#   o → f ✗  o → d ✗  o → o ✓   (outros estático)
TRANSITION_MATRIX = [
    [[1, 1, 0],
     [0, 1, 0],
     [0, 0, 1]],
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_shapefile(path: pathlib.Path | str) -> gpd.GeoDataFrame:
    """Aceita .shp direto ou .zip contendo um único .shp."""
    path = pathlib.Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".shp")]
            if not names:
                raise FileNotFoundError(f"Nenhum .shp encontrado em {path}")
            return gpd.read_file(f"zip://{path}!{names[0]}")
    return gpd.read_file(str(path))


def discrete_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """
    Métricas para saídas binárias (0/1).

    Parameters
    ----------
    pred : array de floats/ints {0, 1} — saída Python
    ref  : array de floats/ints {0, 1} — referência TerraME
    """
    pred = (pred >= 0.5).astype(int)
    ref  = (ref  >= 0.5).astype(int)

    n          = len(pred)
    agree      = int((pred == ref).sum())
    tp         = int(((pred == 1) & (ref == 1)).sum())
    tn         = int(((pred == 0) & (ref == 0)).sum())
    fp         = int(((pred == 1) & (ref == 0)).sum())
    fn         = int(((pred == 0) & (ref == 1)).sum())

    p0         = agree / n
    # Cohen's kappa
    p_yes      = ((tp + fp) / n) * ((tp + fn) / n)
    p_no       = ((tn + fn) / n) * ((tn + fp) / n)
    pe         = p_yes + p_no
    kappa      = (p0 - pe) / (1 - pe) if pe < 1 else 1.0

    precision  = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall     = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else float("nan"))

    return dict(
        n=n, agree=agree, accuracy=p0 * 100,
        tp=tp, tn=tn, fp=fp, fn=fn,
        kappa=kappa, precision=precision, recall=recall, f1=f1,
    )


# ── simulation ────────────────────────────────────────────────────────────────

def run_python(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, float]:
    """Executa a simulação Python e retorna (gdf_result, ms_por_passo)."""
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


# ── alignment ─────────────────────────────────────────────────────────────────

def align_on_col_lin(
    gdf_py:      gpd.GeoDataFrame,
    gdf_terrame: gpd.GeoDataFrame,
    col_col:     str = "col",
    lin_col:     str = "lin",
) -> pd.DataFrame:
    """
    Alinha os dois GDFs pelo índice (col, lin) — equivalente ao (row, col)
    do validate_lab1.py. O TerraME salva essas colunas no shapefile de saída.

    Retorna DataFrame com colunas: d_py, d_terrame.
    """
    idx_py  = pd.MultiIndex.from_arrays(
        [gdf_py[col_col].astype(int),  gdf_py[lin_col].astype(int)],
        names=["col", "lin"],
    )
    idx_ter = pd.MultiIndex.from_arrays(
        [gdf_terrame[col_col].astype(int), gdf_terrame[lin_col].astype(int)],
        names=["col", "lin"],
    )

    s_py  = pd.Series(gdf_py["d"].values,        index=idx_py,  name="d_py")
    # TerraME salva d_out; aceita também d caso a coluna seja renomeada
    ter_col = "d_out" if "d_out" in gdf_terrame.columns else "d"
    s_ter = pd.Series(gdf_terrame[ter_col].values, index=idx_ter, name="d_terrame")

    return s_py.to_frame().join(s_ter, how="inner")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_scatter(df: pd.DataFrame, m: dict, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    jitter  = np.random.default_rng(42).uniform(-0.02, 0.02, len(df))
    ax.scatter(df["d_terrame"] + jitter, df["d_py"] + jitter,
               alpha=0.15, s=5, color="steelblue")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("TerraME d_out (2004)")
    ax.set_ylabel("Python d (2004)")
    ax.set_title("Lab6 — d (desmatamento) 2004\nPython vs TerraME")
    ax.text(
        0.05, 0.88,
        f"acc={m['accuracy']:.2f}%  κ={m['kappa']:.4f}\n"
        f"F1={m['f1']:.4f}  TP={m['tp']}  FP={m['fp']}\n"
        f"FN={m['fn']}  TN={m['tn']}  N={m['n']}",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout()
    plt.savefig(str(out), dpi=150)
    print(f"Plot: {out}")


def plot_map(gdf: gpd.GeoDataFrame, out: pathlib.Path) -> None:
    """
    Mapa espacial de concordância: verde = concorda, vermelho = discorda.
    Requer coluna 'agree' no GDF (1 = concorda, 0 = discorda).
    """
    if "agree" not in gdf.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf[gdf["agree"] == 1].plot(ax=ax, color="green",  markersize=1, label="concorda")
    gdf[gdf["agree"] == 0].plot(ax=ax, color="red",    markersize=2, label="discorda")
    ax.set_title("Lab6 — concordância espacial d (2004)\nPython vs TerraME")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out), dpi=150)
    print(f"Map: {out}")


# ── report ────────────────────────────────────────────────────────────────────

def write_report(m: dict, ms: float, n_total: int, out: pathlib.Path) -> None:
    demand_final = ANNUAL_DEMAND[-1]
    lines = [
        "# Lab6 Validation Report — CLUE-S Discrete\n\n",
        f"Grid: cs_moju | Cells: {n_total} | Steps: {N_STEPS} (1999–2004)\n\n",
        "## Runtime\n\n",
        f"| ms/step |\n|---|\n| {ms:.1f} |\n\n",
        "## Demand check (step 5 = 2004)\n\n",
        "| LU | Demand | Note |\n|---|---|---|\n",
        f"| f  | {demand_final[0]} | floresta |\n",
        f"| d  | {demand_final[1]} | desmatamento |\n",
        f"| o  | {demand_final[2]} | outros (estático) |\n\n",
        "## Accuracy — `d` at step 5 (2004)\n\n",
        "| Metric | Value |\n|---|---|\n",
        f"| Overall Accuracy | {m['accuracy']:.4f}% |\n",
        f"| Cohen's κ        | {m['kappa']:.4f} |\n",
        f"| Precision (d=1)  | {m['precision']:.4f} |\n",
        f"| Recall (d=1)     | {m['recall']:.4f} |\n",
        f"| F1 Score         | {m['f1']:.4f} |\n",
        f"| TP               | {m['tp']} |\n",
        f"| TN               | {m['tn']} |\n",
        f"| FP               | {m['fp']} |\n",
        f"| FN               | {m['fn']} |\n",
        f"| N (aligned)      | {m['n']} |\n",
    ]
    out.write_text("".join(lines))
    print(f"Report: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida disslucc-discrete (Lab6) contra saída TerraME."
    )
    parser.add_argument("cs_moju",  help="cs_moju.shp ou cs_moju.zip (dado de entrada)")
    parser.add_argument("terrame",  help="Shapefile gerado pelo TerraME (Lab6_2004.shp ou .zip)")
    parser.add_argument(
        "--col-attr", default="col",
        help="Nome da coluna de coluna no shapefile (default: col)",
    )
    parser.add_argument(
        "--lin-attr", default="lin",
        help="Nome da coluna de linha no shapefile (default: lin)",
    )
    parser.add_argument(
        "--save-shp", default="lab6_python_2004.shp",
        help="Caminho para salvar o shapefile resultado do Python",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Lab6 Validation: Python (disslucc-discrete) vs TerraME")
    print("=" * 60)

    # ── load data ─────────────────────────────────────────────────────────────
    print("\n[1/3] Carregando dados...")
    gdf_input   = load_shapefile(args.cs_moju)
    gdf_terrame = load_shapefile(args.terrame)
    print(f"  Entrada:  {len(gdf_input)} células  crs={gdf_input.crs}")
    ter_col = "d_out" if "d_out" in gdf_terrame.columns else "d"
    print(f"  TerraME:  {len(gdf_terrame)} células  coluna de referência='{ter_col}'")

    # sanity check: soma f+d+o deve ser 1 para todas as células
    total = gdf_input["f"] + gdf_input["d"] + gdf_input["o"]
    if not (np.abs(total - 1.0) < 1e-6).all():
        n_bad = int((np.abs(total - 1.0) >= 1e-6).sum())
        print(f"  AVISO: {n_bad} células com f+d+o ≠ 1 no dado de entrada")

    # ── run python simulation ─────────────────────────────────────────────────
    print("\n[2/3] Executando simulação Python...")
    gdf_result, ms = run_python(gdf_input)
    print(f"  {ms:.1f} ms/passo")

    # sanity check: demanda alcançada no último passo
    for i, lu in enumerate(LAND_USE_TYPES):
        n    = int((gdf_result[lu] == 1).sum())
        dem  = ANNUAL_DEMAND[-1][i]
        diff = n - dem
        flag = " ✓" if abs(diff) <= 10 else f" ✗ (diff={diff:+d})"
        print(f"  {lu}: alocado={n}  demanda={dem}{flag}")

    # ── compare ───────────────────────────────────────────────────────────────
    print("\n[3/3] Comparando...")
    df_aligned = align_on_col_lin(
        gdf_result, gdf_terrame,
        col_col=args.col_attr, lin_col=args.lin_attr,
    )
    print(f"  Células alinhadas: {len(df_aligned)} / {len(gdf_input)}")

    m = discrete_metrics(df_aligned["d_py"].values, df_aligned["d_terrame"].values)
    print(f"  Accuracy:  {m['accuracy']:.4f}%")
    print(f"  κ (kappa): {m['kappa']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")

    # ── save comparison shapefile ─────────────────────────────────────────────
    # Adiciona coluna 'agree': 1 = concorda com TerraME, 0 = discorda
    agree_series = pd.Series(
        (df_aligned["d_py"].values >= 0.5).astype(int)
        == (df_aligned["d_terrame"].values >= 0.5).astype(int),
        index=df_aligned.index,
    ).astype(int)

    # Mapeia de volta para as linhas do gdf_result pelo índice (col, lin)
    idx_result = pd.MultiIndex.from_arrays(
        [gdf_result[args.col_attr].astype(int), gdf_result[args.lin_attr].astype(int)],
        names=["col", "lin"],
    )
    gdf_result["agree"] = agree_series.reindex(idx_result).values

    out_shp = pathlib.Path(args.save_shp)
    cols_to_save = ["f", "d", "o", "agree", "geometry"]
    gdf_result[cols_to_save].to_file(str(out_shp))
    print(f"\nShapefile salvo: {out_shp}")

    # ── outputs ───────────────────────────────────────────────────────────────
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
