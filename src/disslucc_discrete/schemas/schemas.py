"""
disslucc_discrete.schemas
-------------------------
Dataclasses que definem os parâmetros de cada componente.

Servem como contratos entre o usuário e os modelos — validação
acontece aqui, os modelos só consomem.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LogisticRegressionSpec:
    """
    Parâmetros de uma regressão logística para um uso do solo.

    Usado por PotentialDLogisticRegression (CLUE-S discreto).
    Traduz o bloco de parâmetros de PotentialDLogisticRegression.lua:

        {
          const       = -2.34,
          elasticity  = 0.6,
          betas       = { dist_br = -2.51, ... }
        }
    """

    const: float
    elasticity: float = 0.0
    betas: dict[str, float] = field(default_factory=dict)


