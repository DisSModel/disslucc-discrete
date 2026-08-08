"""
disslucc_discrete.components.potential.logistic_regression
---------------------------------------------------------
Potencial por regressão logística — CLUE-S discreto.
Translation of PotentialDLogisticRegression.lua (LuccME / TerraME).
"""

# fixed: updated import to disslucc_discrete.schemas.schemas
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from dissmodel.geo import SyncSpatialModel

from disslucc_discrete.schemas.schemas import LogisticRegressionSpec

if TYPE_CHECKING:
    import pandas as pd


class PotentialDLogisticRegression(SyncSpatialModel):
    """
    Transition potential via logistic regression (CLUE-S).

    Verburg et al. (2002). Para cada célula e cada uso do solo:

        z    = const + Σ(beta_k × cell[k])
        prob = e^z / (1 + e^z)           # logistic / sigmoid
        pot  = prob + elasticity × I(cell[lu] == 1)

    A elasticidade reforça o uso corrente da célula: quanto mais próxima
    to 1, the harder the transition to another use (spatial lock-in).

    Parameters
    ----------
    potential_data : list[list[LogisticRegressionSpec]]
        potential_data[region_idx][lu_idx] — regiões × usos do solo.
        region_idx is 0-based; region 1 in the TOML model is index 0.
    land_use_types : list[str]
        Nomes dos usos do solo na mesma ordem que potential_data[r].
    region_attr : str
        Nome da coluna de região no GeoDataFrame (default "region").
        If absent, it is created with value 1 (a single region).

    Columns written
    ---------------
    {lu}_reg : float   — probabilidade logística bruta (sem elasticidade)
    {lu}_pot : float   — probability + elasticity (used by allocation)
    """

    def setup(
        self,
        potential_data: list[list[LogisticRegressionSpec]],
        land_use_types: list[str],
        region_attr: str = "region",
    ) -> None:
        self.potential_data = potential_data
        self.land_use_types = land_use_types
        self.region_attr = region_attr

        # Garante coluna de região — equivale ao cell.region = 1 do Lua
        if self.region_attr not in self.gdf.columns:
            self.gdf[self.region_attr] = 1

        # Inicializa colunas de saída
        for lu in self.land_use_types:
            self.gdf[lu + "_reg"] = 0.0
            self.gdf[lu + "_pot"] = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def execute(self) -> None:
        """Recalcula potencial para todas as regiões e usos do solo."""
        for r_idx, region_specs in enumerate(self.potential_data):
            r_number = r_idx + 1
            mask = self.gdf[self.region_attr] == r_number
            for lu_idx, spec in enumerate(region_specs):
                self._compute_potential(mask, lu_idx, spec)

    # ── internals ─────────────────────────────────────────────────────────────

    def _compute_potential(
        self,
        mask: "pd.Series[bool]",
        lu_idx: int,
        spec: LogisticRegressionSpec,
    ) -> None:
        """
        Calcula _reg e _pot para um uso do solo em uma região.

        Traduz exatamente calcRegressionLogistic + probability + elasticity
        do PotentialDLogisticRegression.lua.
        """
        lu = self.land_use_types[lu_idx]

        # Step 1: linear combination  z = const + sum(beta_k * x_k)
        # Inicializa com const; multiplica por zero para herdar o índice do GDF.
        z = self.gdf.loc[mask, self.land_use_types[0]] * 0.0 + spec.const
        for col, beta in spec.betas.items():
            z = z + beta * self.gdf.loc[mask, col]

        # Step 2: logistic transformation  prob = sigmoid(z)
        # np.exp(-z) is numerically more stable than e^z / (1 + e^z)
        # para z muito negativo, e idêntico a e^z/(1+e^z) para outros valores.
        prob = 1.0 / (1.0 + np.exp(-z))

        # Step 3: elasticity for the current land use
        elas = np.where(self.gdf.loc[mask, lu] == 1, spec.elasticity, 0.0)

        self.gdf.loc[mask, lu + "_reg"] = prob
        self.gdf.loc[mask, lu + "_pot"] = prob + elas
