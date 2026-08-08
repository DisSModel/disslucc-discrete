"""
disslucc_discrete.executors.clue_s_vector_executor
--------------------------------------------------
Executor para simulações LUCC vetoriais discretas (CLUE-S / GeoDataFrame).
Equivalente ao lab15_main.lua — funciona via CLI e API da plataforma.

Differences from LUCCVectorExecutor (continuous CLUE)
------------------------------------------------------------
- Usa PotentialDLogisticRegression  em vez de PotentialLinearRegression
- Usa AllocationDClueSLike          em vez de AllocationClueLike
- Células são binárias (0/1 por uso) em vez de contínuas [0, 1]
- Demand as cell counts               instead of area (when cell_area = 1)
- Sem complementar_lu / correctCellChange
"""

from __future__ import annotations

import geopandas as gpd
from dissmodel.executor import ExperimentRecord, ModelExecutor
from dissmodel.executor.cli import run_cli
from dissmodel.io import load_dataset, save_dataset

from disslucc_discrete.common.utils import default_output_uri


class ClueSVectorExecutor(ModelExecutor):
    """
    Executor CLUE-S discreto para substrato vetorial.

    Contrato do record
    ------------------
    record.source.uri
        Caminho para o GeoDataFrame de entrada (GeoPackage, Shapefile, etc.).

    record.parameters
        n_steps    : int   — number of time steps (default 6)
        cell_area  : float — área de cada célula nas unidades da demanda (default 1.0)
        demand_csv : str   — URI do CSV de demanda (opcional; tem prioridade sobre
                             annual_demand quando ambos estão presentes)

    record.resolved_spec["model"]
        land_use_types : list[str]
            Usos do solo na mesma ordem das colunas binárias do GDF
            e das linhas/colunas da transition_matrix.
        potential : list[dict]
            Um dict por uso do solo (na ordem de land_use_types):
              { "const": float, "elasticity": float, "betas": {col: float} }
        transition_matrix : list[list[list[int]]]
            [region_idx][from_lu][to_lu] in {0, 1}. region_idx is 0-based.
        allocation : dict
            { "max_difference": float, "max_iteration": int,
              "factor_iteration": float }
        annual_demand : list[list[float]]
            [step][lu_idx] — used when demand_csv is absent.
        region_attr : str   (optional, default "region")
        cell_area   : float (optional; sobreposto por parameters.cell_area)
    """

    name = "clue_s_vector"

    # ── public contract ───────────────────────────────────────────────────────

    def load(self, record: ExperimentRecord) -> gpd.GeoDataFrame:
        gdf, checksum = load_dataset(record.source.uri)
        record.source.checksum = checksum

        if record.column_map:
            gdf = gdf.rename(columns={v: k for k, v in record.column_map.items()})

        record.add_log(f"Carregado GDF: {len(gdf)} feições  crs={gdf.crs}")
        return gdf

    def validate(self, record: ExperimentRecord) -> None:
        """
        Verificações estáticas no record antes de carregar dados.
        Erros de coluna são detectados em run() após o load().
        """
        spec = record.resolved_spec.get("model", {})
        lu_types = spec.get("land_use_types", [])

        if not lu_types:
            raise ValueError("model.land_use_types está ausente no spec.")

        tm = spec.get("transition_matrix", {}).get("data") or spec.get(
            "transition_matrix"
        )
        if tm is None:
            raise ValueError("model.transition_matrix está ausente no spec.")

        n_lu = len(lu_types)
        for r_idx, region in enumerate(tm):
            if len(region) != n_lu:
                raise ValueError(
                    f"transition_matrix região {r_idx}: esperava {n_lu} linhas, "
                    f"encontrou {len(region)}."
                )
            for row in region:
                if len(row) != n_lu:
                    raise ValueError(
                        f"transition_matrix região {r_idx}: linha com {len(row)} "
                        f"colunas, esperava {n_lu}."
                    )
                if any(v not in (0, 1) for v in row):
                    raise ValueError(
                        "transition_matrix contém valores inválidos "
                        f"(apenas 0 ou 1 são aceitos): {row}"
                    )

        pot = spec.get("potential", [])
        if len(pot) != n_lu:
            raise ValueError(
                f"model.potential tem {len(pot)} entradas; "
                f"esperava {n_lu} (uma por land_use_type)."
            )

    def run(self, data: gpd.GeoDataFrame, record: ExperimentRecord) -> gpd.GeoDataFrame:
        """
        Validate columns and run the discrete CLUE-S simulation.

        `data` is the GeoDataFrame injected by execute_lifecycle — no I/O here.
        """
        from dissmodel.core import Environment

        from disslucc_discrete import DemandPreComputedValues, load_demand_csv
        from disslucc_discrete.components.allocation.vector.clue_s import AllocationDClueSLike
        from disslucc_discrete.components.potential.vector.logistic_regression import (
            PotentialDLogisticRegression,
        )
        from disslucc_discrete.schemas.schemas import LogisticRegressionSpec

        spec = record.resolved_spec.get("model", {})
        params = record.parameters
        lu_types = spec.get("land_use_types", ["f", "d", "o"])
        n_steps = params.get("n_steps", 6)

        gdf = data
        _check_columns(gdf, spec, lu_types)

        # ── transition matrix ─────────────────────────────────────────────
        tm_raw = spec.get("transition_matrix", {})
        tm = tm_raw.get("data", tm_raw) if isinstance(tm_raw, dict) else tm_raw

        # ── demanda ───────────────────────────────────────────────────────
        if "demand_csv" in params:
            from dissmodel.io._utils import read_text

            raw_csv = read_text(params["demand_csv"])
            annual_demand = load_demand_csv(raw_csv, lu_types)
        else:
            raw = spec.get("annual_demand", [])
            annual_demand = [
                entry.get("values", entry) if isinstance(entry, dict) else entry
                for entry in raw
            ]

        if len(annual_demand) < n_steps:
            raise ValueError(
                f"annual_demand tem {len(annual_demand)} entradas para "
                f"{n_steps} time steps."
            )

        # ── allocation parameters ─────────────────────────────────────────
        alloc_cfg = spec.get("allocation", {})
        cell_area = float(params.get("cell_area") or spec.get("cell_area", 1.0))
        # ── environment + modelos ─────────────────────────────────────────
        env = Environment(end_time=n_steps - 1)  # inclusive end_time: runs steps 0..n_steps-1

        demand = DemandPreComputedValues(
            annual_demand=annual_demand,
            land_use_types=lu_types,
        )

        potential_specs = [
            LogisticRegressionSpec(
                const=p["const"],
                elasticity=p.get("elasticity", 0.0),
                betas=p.get("betas", {}),
            )
            for p in spec.get("potential", [])
        ]

        PotentialDLogisticRegression(
            gdf=gdf,
            potential_data=[potential_specs],
            land_use_types=lu_types,
            region_attr=spec.get("region_attr", "region"),
        )

        AllocationDClueSLike(
            gdf=gdf,
            demand=demand,
            land_use_types=lu_types,
            transition_matrix=tm,
            cell_area=cell_area,
            max_difference=alloc_cfg.get("max_difference", 10.0),
            max_iteration=alloc_cfg.get("max_iteration", 2000),
            factor_iteration=alloc_cfg.get("factor_iteration", 0.0001),
            region_attr=spec.get("region_attr", "region"),
        )

        if params.get("interactive", False):
            from dissmodel.visualization import Map

            Map(
                gdf=gdf,
                plot_params={
                    "column": lu_types[0],
                    "cmap": "YlGn",
                    "legend": True,
                },
            )

        record.add_log(
            f"Starting discrete CLUE-S simulation: "
            f"{n_steps} steps · {len(gdf)} cells · {len(lu_types)} land uses"
        )
        env.run()

        if params.get("interactive", False):
            import matplotlib.pyplot as plt

            plt.show()

        record.add_log("Simulation complete.")
        return gdf

    def save(
        self, result: gpd.GeoDataFrame, record: ExperimentRecord
    ) -> ExperimentRecord:
        uri = record.output_path or default_output_uri(record.experiment_id, ext="gpkg")
        checksum = save_dataset(result, uri)

        record.output_path = uri
        record.output_sha256 = checksum
        record.status = "completed"
        record.add_log(f"Salvo em {uri}")
        return record


def _check_columns(
    gdf: gpd.GeoDataFrame,
    spec: dict,
    lu_types: list[str],
) -> None:
    driver_cols: set[str] = set()
    for p in spec.get("potential", []):
        driver_cols.update(p.get("betas", {}).keys())

    expected = set(lu_types) | driver_cols
    missing = expected - set(gdf.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes após column_map: {missing}\n"
            f"Colunas disponíveis: {sorted(gdf.columns)}"
        )


if __name__ == "__main__":
    run_cli(ClueSVectorExecutor)
