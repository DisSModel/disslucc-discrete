"""
disslucc_discrete.components.allocation.clue_s
----------------------------------------------
Discrete CLUE-S-like allocation (Verburg et al. 2002).
Translation of AllocationDClueSLike.lua (LuccME / TerraME).

Each cell holds exactly one land use (binary 0/1 columns). Allocation
iteratively adjusts a global correction vector per land use until the
difference between demand and allocated area falls within `max_difference`.
"""

from __future__ import annotations

import numpy as np
from dissmodel.geo import SyncSpatialModel


class AllocationDClueSLike(SyncSpatialModel):
    """
    Discrete CLUE-S allocation — cell-by-cell competition.

    Algorithm (per time step)
    --------------------------------
    1. iter_vec[lu] = 0  for every lu  (reset at each step)
    2. For each cell: best_lu = argmax { (1 + tau_lu) × pot_lu + iter_lu },
       restricted to the transitions allowed by transition_matrix.
    3. Compute diff[lu] = demand[lu] − allocated_area[lu].
    4. iter_vec[lu] += diff[lu] × factor_iteration.
    5. Repeat 2–4 until max(|diff[lu]|) <= max_difference or n_iter >= max_iteration.

    Parameters
    ----------
    demand : DemandProtocol
        Demand component. Must expose get_current_lu_demand(i).
    land_use_types : list[str]
        Land-use names, in the order of the GDF's binary columns.
    transition_matrix : list[list[list[int]]]
        transition_matrix[region_idx][from_lu][to_lu] in {0, 1}.
        region_idx is 0-based; 1 = transition allowed, 0 = forbidden.
    cell_area : float
        Area of each cell, in the same units as the demand.
        Use 1.0 when demand is expressed as a number of cells.
    max_difference : float
        Stopping criterion: largest acceptable absolute difference between
        demand and allocated area (in units of cell_area × n_cells).
    max_iteration : int
        Inner-loop iteration limit before raising RuntimeError.
    factor_iteration : float
        Adjustment rate of the iteration vector, equivalent to the Lua
        factorIteration. Smaller values converge more smoothly but slower.
    region_attr : str
        Region column in the GDF (default "region"). Created as 1 if absent.

    Tau columns (optional)
    ----------------------
    If the GDF has `tau_{lu}` columns, they are used as a per-cell
    attraction/repulsion factor (an optional LuccME feature). If absent,
    tau = 0 for every cell (the Lab15 default).
    """

    def setup(
        self,
        demand,
        land_use_types: list[str],
        transition_matrix: list[list[list[int]]],
        cell_area: float = 1.0,
        max_difference: float = 10.0,
        max_iteration: int = 2000,
        factor_iteration: float = 0.0001,
        region_attr: str = "region",
    ) -> None:
        self.demand = demand
        self.land_use_types = land_use_types
        self.cell_area = cell_area
        self.max_difference = max_difference
        self.max_iteration = max_iteration
        self.factor_iteration = factor_iteration
        self.region_attr = region_attr

        # Pre-compila a transition_matrix como array NumPy:
        # shape (n_regions, n_lu, n_lu) — acesso O(1) no inner loop
        self._tm = np.array(transition_matrix, dtype=np.int8)

        if self.region_attr not in self.gdf.columns:
            self.gdf[self.region_attr] = 1

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def execute(self) -> None:
        """
        Run the CLUE-S inner loop for the current time step.
        Equivale ao bloco while do AllocationDClueSLike.run() no Lua.
        """
        lu_types = self.land_use_types
        n_lu = len(lu_types)

        # Per-cell regions (0-based for numpy indexing)
        regions = self.gdf[self.region_attr].values.astype(int) - 1

        # ── ESTADO INICIAL DO PASSO ───────────────────────────────────────
        # The transition matrix must be based on the state at the start of the step,
        # permitindo que o algoritmo oscile entre usos permitidos durante
        # a busca pelo equilíbrio (convergência).
        lu_matrix_start = np.column_stack([self.gdf[lu].values for lu in lu_types])
        initial_lu_idx = np.argmax(lu_matrix_start, axis=1)  # (n_cells,)
        allowed = self._tm[regions, initial_lu_idx, :]  # (n_cells, n_lu)

        # Tau por célula e uso: (n_cells, n_lu)
        # tau_{lu} column is optional — default 0 (no attraction/repulsion)
        tau = np.zeros((len(self.gdf), n_lu), dtype=float)
        for j, lu in enumerate(lu_types):
            col = f"tau_{lu}"
            if col in self.gdf.columns:
                tau[:, j] = self.gdf[col].values

        # Iteration vector — reset at each time step
        iter_vec = np.zeros(n_lu, dtype=float)

        for n_iter in range(self.max_iteration + 1):

            # ── 1. Escore por célula e uso ────────────────────────────────
            # score = (1 + tau) × pot + iter_vec
            pot = np.column_stack([self.gdf[lu + "_pot"].values for lu in lu_types])
            scores = (1.0 + tau) * pot + iter_vec[np.newaxis, :]

            # Transições proibidas recebem -inf para nunca vencer o argmax
            scores = np.where(allowed, scores, -np.inf)

            # ── 2. Melhor uso por célula ──────────────────────────────────
            best_lu_idx = np.argmax(scores, axis=1)  # (n_cells,)

            for j, lu in enumerate(lu_types):
                self.gdf[lu] = (best_lu_idx == j).astype(float)

            # ── 3. Verifica convergência ──────────────────────────────────
            diff = self._calc_diff()
            max_diff = float(np.max(np.abs(list(diff.values()))))

            if max_diff <= self.max_difference:
                break

            if n_iter >= self.max_iteration:
                raise RuntimeError(
                    f"Allocation did not converge at step {int(self.env.now())} "
                    f"after {self.max_iteration} iterations "
                    f"(max error = {max_diff:.2f})"
                )

            # ── 4. Adjust iteration vector ────────────────────────────────
            for j, lu in enumerate(lu_types):
                iter_vec[j] += diff[lu] * self.factor_iteration

    # ── helpers ───────────────────────────────────────────────────────────────

    def _calc_diff(self) -> dict[str, float]:
        """
        Diferença entre demanda e área alocada para cada uso do solo.
        Equivale a calcDifferences() do Lua.

        diff[lu] = demand[lu] − (n_cells_with_lu × cell_area)
        """
        return {
            lu: (
                self.demand.get_current_lu_demand(i)
                - float((self.gdf[lu] == 1).sum()) * self.cell_area
            )
            for i, lu in enumerate(self.land_use_types)
        }
