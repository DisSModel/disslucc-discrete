"""
disslucc.executors.clue_s_vector_executor
------------------------------------------
Executor para simulações LUCC vetoriais discretas (CLUE-S / GeoDataFrame).
Equivalente ao lab6_main.lua — funciona via CLI e API da plataforma.

Diferenças em relação ao LUCCVectorExecutor (CLUE contínuo)
------------------------------------------------------------
- Usa PotentialDLogisticRegression  em vez de PotentialLinearRegression
- Usa AllocationDClueSLike          em vez de AllocationClueLike
- Células são binárias (0/1 por uso) em vez de contínuas [0, 1]
- Demanda em contagem de células     em vez de área (quando cell_area = 1)
- Sem complementar_lu / correctCellChange
"""
# fixed: updated internal imports to disslucc.* namespace
from __future__ import annotations

import geopandas as gpd

from dissmodel.executor     import ExperimentRecord, ModelExecutor
from dissmodel.executor.cli import run_cli
from dissmodel.io           import load_dataset, save_dataset

from disslucc.common.utils import default_output_uri


class ClueSVectorExecutor(ModelExecutor):
    """
    Executor CLUE-S discreto para substrato vetorial.

    Contrato do record
    ------------------
    record.source.uri
        Caminho para o GeoDataFrame de entrada (GeoPackage, Shapefile, etc.).

    record.parameters
        n_steps    : int   — número de passos de tempo (default 6)
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
            [region_idx][from_lu][to_lu] ∈ {0, 1}. region_idx é 0-based.
        allocation : dict
            { "max_difference": float, "max_iteration": int,
              "factor_iteration": float }
        annual_demand : list[list[float]]
            [step][lu_idx] — usado quando demand_csv não está presente.
        region_attr : str   (optional, default "region")
        cell_area   : float (optional; sobreposto por parameters.cell_area)

    Exemplo de model.toml
    ----------------------
    [model]
    land_use_types     = ["f", "d", "o"]
    region_attr        = "region"
    cell_area          = 1.0

    [[model.potential]]
    const       = -2.34187976925989
    elasticity  = 0.0
    [model.potential.betas]
    media_decl  = -0.0272710076327129
    dist_area_  = 4.30977432375496
    dist_br     = 3.10319957497883
    dist_curua  = 0.445414024051873
    dist_rios_  = 47.3556329553235
    dist_estra  = 38.4966894254506

    [[model.potential]]
    const       = -0.100351497277102
    elasticity  = 0.6
    [model.potential.betas]
    media_decl  = 0.0581358851690861
    dist_area_  = -0.974998890251365
    dist_br     = -2.51650696123426
    dist_curua  = -1.26742746441679
    dist_rios_  = -40.3646901047482
    dist_estra  = -23.0841140199094

    [[model.potential]]
    const       = 0.01
    elasticity  = 0.5

    [model.allocation]
    max_difference   = 10.0
    max_iteration    = 1000
    factor_iteration = 0.0001

    [model.transition_matrix]
    # region 1 (0-based index 0)
    # rows = from_lu, cols = to_lu  (f, d, o)
    data = [[[1,1,0],[0,1,0],[0,0,1]]]

    [[model.annual_demand]]
    # step 0 → 1999
    values = [5706, 205, 3]
    ...
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
        spec     = record.resolved_spec.get("model", {})
        lu_types = spec.get("land_use_types", [])

        if not lu_types:
            raise ValueError("model.land_use_types está ausente no spec.")

        tm = spec.get("transition_matrix", {}).get("data") or spec.get("transition_matrix")
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
        Valida colunas e executa a simulação CLUE-S discreta.

        `data` é o GeoDataFrame injetado pelo execute_lifecycle — sem I/O aqui.
        """
        from dissmodel.core import Environment
        from disslucc import DemandPreComputedValues, load_demand_csv
        from disslucc.components.potential.logistic  import PotentialDLogisticRegression
        from disslucc.components.allocation.discrete import AllocationDClueSLike
        from disslucc.schemas.schemas                import LogisticRegressionSpec

        spec     = record.resolved_spec.get("model", {})
        params   = record.parameters
        lu_types = spec.get("land_use_types", ["f", "d", "o"])
        n_steps  = params.get("n_steps", 6)

        gdf = data
        _check_columns(gdf, spec, lu_types)

        # ── transition matrix ─────────────────────────────────────────────
        tm_raw = spec.get("transition_matrix", {})
        tm     = tm_raw.get("data", tm_raw) if isinstance(tm_raw, dict) else tm_raw

        # ── demanda ───────────────────────────────────────────────────────
        # Prioridade: demand_csv no params > annual_demand no spec
        if "demand_csv" in params:
            from dissmodel.io._utils import read_text
            raw_csv      = read_text(params["demand_csv"])
            annual_demand = load_demand_csv(raw_csv, lu_types)
        else:
            # annual_demand no spec: lista de listas [[f0,d0,o0], [f1,d1,o1], ...]
            raw = spec.get("annual_demand", [])
            # suporta formato {values: [...]} e formato direto [float, ...]
            annual_demand = [
                entry.get("values", entry) if isinstance(entry, dict) else entry
                for entry in raw
            ]

        if len(annual_demand) < n_steps:
            raise ValueError(
                f"annual_demand tem {len(annual_demand)} entradas para "
                f"{n_steps} passos de tempo."
            )

        # ── parâmetros de alocação ────────────────────────────────────────
        alloc_cfg = spec.get("allocation", {})
        cell_area = float(
            params.get("cell_area") or spec.get("cell_area", 1.0)
        )
        # ── environment + modelos ─────────────────────────────────────────
        # Para executar 'n_steps' (ex: 6 passos de 0 a 5 inclusive),
        # end_time deve ser igual a n_steps, pois o loop é while now < end_time
        env = Environment(end_time=n_steps)

        demand = DemandPreComputedValues(
            annual_demand  = annual_demand,
            land_use_types = lu_types,
        )

        # potential_data: lista de especificações para a região 1 (único região
        # do modelo Lab6). Para modelos multiregião, cada região é uma lista
        # adicional dentro de potential_data.
        from disslucc.schemas.schemas import LogisticRegressionSpec
        potential_specs = [
            LogisticRegressionSpec(
                const      = p["const"],
                elasticity = p.get("elasticity", 0.0),
                betas      = p.get("betas", {}),
            )
            for p in spec.get("potential", [])
        ]


        potential = PotentialDLogisticRegression(
            gdf            = gdf,
            potential_data = [potential_specs],   # região 1
            land_use_types = lu_types,
            region_attr    = spec.get("region_attr", "region"),
        )

        AllocationDClueSLike(
            gdf               = gdf,
            demand            = demand,
            land_use_types    = lu_types,
            transition_matrix = tm,
            cell_area         = cell_area,
            max_difference    = alloc_cfg.get("max_difference",   10.0),
            max_iteration     = alloc_cfg.get("max_iteration",   1000),
            factor_iteration  = alloc_cfg.get("factor_iteration", 0.0001),
            region_attr       = spec.get("region_attr", "region"),
        )

        if params.get("interactive", False):
            from dissmodel.visualization import Map
            Map(
                gdf         = gdf,
                plot_params = {
                    "column": lu_types[0],
                    "cmap":   "YlGn",
                    "legend": True,
                },
            )

        record.add_log(
            f"Iniciando simulação CLUE-S discreta: "
            f"{n_steps} passos · {len(gdf)} células · {len(lu_types)} usos"
        )
        env.run()

        if params.get("interactive", False):
            import matplotlib.pyplot as plt
            plt.show()

        record.add_log("Simulação concluída.")
        return gdf

    def save(self, result: gpd.GeoDataFrame, record: ExperimentRecord) -> ExperimentRecord:
        uri      = record.output_path or default_output_uri(record.experiment_id, ext="gpkg")
        checksum = save_dataset(result, uri)

        record.output_path   = uri
        record.output_sha256 = checksum
        record.status        = "completed"
        record.add_log(f"Salvo em {uri}")
        return record


# ── helpers ───────────────────────────────────────────────────────────────────


def _check_columns(
    gdf:      gpd.GeoDataFrame,
    spec:     dict,
    lu_types: list[str],
) -> None:
    """
    Verifica colunas obrigatórias após aplicação do column_map.
    Executado dentro de run() onde o GDF já foi carregado.
    """
    # Colunas de uso do solo (binárias 0/1)
    driver_cols: set[str] = set()
    for p in spec.get("potential", []):
        driver_cols.update(p.get("betas", {}).keys())

    expected = set(lu_types) | driver_cols
    missing  = expected - set(gdf.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes após column_map: {missing}\n"
            f"Colunas disponíveis: {sorted(gdf.columns)}\n"
            "Verifique column_map ou betas no model.toml."
        )


if __name__ == "__main__":
    run_cli(ClueSVectorExecutor)
