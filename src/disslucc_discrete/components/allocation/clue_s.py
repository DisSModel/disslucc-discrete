"""
disslucc_discrete.components.allocation.clue_s
----------------------------------------------
Alocação discreta tipo CLUE-S (Verburg et al. 2002).
Tradução de AllocationDClueSLike.lua (LuccME / TerraME).

Cada célula possui exatamente um uso do solo (colunas binárias 0/1).
A alocação ajusta iterativamente um vetor de correção global por uso
até que a diferença entre demanda e área alocada esteja dentro de
`max_difference`.
"""
from __future__ import annotations

import numpy as np

from dissmodel.geo import SyncSpatialModel


class AllocationDClueSLike(SyncSpatialModel):
    """
    Alocação discreta CLUE-S — competição célula a célula.

    Algoritmo (por passo de tempo)
    --------------------------------
    1. iter_vec[lu] = 0  para todo lu  (reinicializado a cada passo)
    2. Para cada célula: best_lu = argmax { (1 + tau_lu) × pot_lu + iter_lu }
       restrito às transições permitidas pela transition_matrix.
    3. Calcula diff[lu] = demand[lu] − area_alocada[lu].
    4. iter_vec[lu] += diff[lu] × factor_iteration.
    5. Repete 2–4 até max(|diff[lu]|) ≤ max_difference ou n_iter ≥ max_iteration.

    Parameters
    ----------
    demand : DemandProtocol
        Componente de demanda. Deve expor get_current_lu_demand(i).
    land_use_types : list[str]
        Nomes dos usos do solo na ordem das colunas binárias do GDF.
    transition_matrix : list[list[list[int]]]
        transition_matrix[region_idx][from_lu][to_lu] ∈ {0, 1}.
        region_idx 0-based; valores 1 = transição permitida, 0 = proibida.
    cell_area : float
        Área de cada célula nas mesmas unidades da demanda.
        Use 1.0 quando a demanda for expressa em número de células.
    max_difference : float
        Critério de parada: máxima diferença absoluta aceitável entre
        demanda e área alocada (nas unidades de cell_area × n_células).
    max_iteration : int
        Limite de iterações do inner loop antes de lançar RuntimeError.
    factor_iteration : float
        Taxa de ajuste do vetor de iteração. Equivale ao factorIteration
        do Lua. Valores menores = convergência mais suave, mais lenta.
    region_attr : str
        Coluna de região no GDF (default "region"). Criada como 1 se ausente.

    Tau columns (opcional)
    ----------------------
    Se o GDF possuir colunas `tau_{lu}`, elas são usadas como fator de
    atração/repulsão por célula (feature opcional do LuccME). Caso ausentes,
    tau = 0 para todas as células (comportamento padrão do Lab6).
    """

    def setup(
        self,
        demand,
        land_use_types:    list[str],
        transition_matrix: list[list[list[int]]],
        cell_area:         float = 1.0,
        max_difference:    float = 10.0,
        max_iteration:     int   = 2000,
        factor_iteration:  float = 0.0001,
        region_attr:       str   = "region",
    ) -> None:
        self.demand            = demand
        self.land_use_types    = land_use_types
        self.cell_area         = cell_area
        self.max_difference    = max_difference
        self.max_iteration     = max_iteration
        self.factor_iteration  = factor_iteration
        self.region_attr       = region_attr

        # Pre-compila a transition_matrix como array NumPy:
        # shape (n_regions, n_lu, n_lu) — acesso O(1) no inner loop
        self._tm = np.array(transition_matrix, dtype=np.int8)

        if self.region_attr not in self.gdf.columns:
            self.gdf[self.region_attr] = 1

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def execute(self) -> None:
        """
        Executa o inner loop CLUE-S para o passo de tempo corrente.
        Equivale ao bloco while do AllocationDClueSLike.run() no Lua.
        """
        lu_types = self.land_use_types
        n_lu     = len(lu_types)

        # Regiões por célula (0-based para indexação do numpy)
        regions = self.gdf[self.region_attr].values.astype(int) - 1

        # ── ESTADO INICIAL DO PASSO ───────────────────────────────────────
        # A matriz de transição deve ser baseada no estado inicial do passo,
        # permitindo que o algoritmo oscile entre usos permitidos durante
        # a busca pelo equilíbrio (convergência).
        lu_matrix_start = np.column_stack([self.gdf[lu].values for lu in lu_types])
        initial_lu_idx  = np.argmax(lu_matrix_start, axis=1)  # (n_cells,)
        allowed         = self._tm[regions, initial_lu_idx, :]  # (n_cells, n_lu)

        # Tau por célula e uso: (n_cells, n_lu)
        # Coluna tau_{lu} é opcional — padrão = 0 (sem atração/repulsão)
        tau = np.zeros((len(self.gdf), n_lu), dtype=float)
        for j, lu in enumerate(lu_types):
            col = f"tau_{lu}"
            if col in self.gdf.columns:
                tau[:, j] = self.gdf[col].values

        # Vetor de iteração — reinicializado a cada passo de tempo
        iter_vec = np.zeros(n_lu, dtype=float)

        for n_iter in range(self.max_iteration + 1):

            # ── 1. Escore por célula e uso ────────────────────────────────
            # score = (1 + tau) × pot + iter_vec
            pot    = np.column_stack([self.gdf[lu + "_pot"].values for lu in lu_types])
            scores = (1.0 + tau) * pot + iter_vec[np.newaxis, :]

            # Transições proibidas recebem -inf para nunca vencer o argmax
            scores = np.where(allowed, scores, -np.inf)

            # ── 2. Melhor uso por célula ──────────────────────────────────
            best_lu_idx = np.argmax(scores, axis=1)   # (n_cells,)

            for j, lu in enumerate(lu_types):
                self.gdf[lu] = (best_lu_idx == j).astype(float)

            # ── 3. Verifica convergência ──────────────────────────────────
            diff     = self._calc_diff()
            max_diff = float(np.max(np.abs(list(diff.values()))))

            if max_diff <= self.max_difference:
                break

            if n_iter >= self.max_iteration:
                raise RuntimeError(
                    f"Alocação não convergiu no passo {int(self.env.now())} "
                    f"após {self.max_iteration} iterações "
                    f"(erro máximo = {max_diff:.2f})"
                )

            # ── 4. Ajusta vetor de iteração ───────────────────────────────
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
