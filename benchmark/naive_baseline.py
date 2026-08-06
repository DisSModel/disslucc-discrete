"""
benchmark/naive_baseline.py — Naive baseline for measuring discriminance
=======================================================================

Why this module exists
----------------------
Reproducing the TerraME output is a **necessary** condition, not a sufficient
one, for claiming that CLUE-S was faithfully translated. If a trivial procedure
— one that does not implement the algorithm in question — reaches the same
parity, then the benchmark is not measuring the algorithm, but something far
simpler.

This module implements that trivial procedure for the Lab6 scenario.

What the baseline does
----------------------
Nothing beyond:

  1. computing ``prob_d`` and ``prob_f`` from the two logistic regressions;
  2. ranking eligible cells by the margin ``prob_d - prob_f``;
  3. marking the top ``N`` as deforested, where ``N`` is the final demand minus
     the cells already deforested in the initial state.

There is no CLUE-S, no iteration vector, no time steps, no transition matrix,
no ``Environment``, and no DisSModel component of any kind.

Why this works on Lab6
----------------------
The scenario collapses into a static threshold:

  * every covariate is static (``dist_br``, ``dist_area_``, ``media_decl`` and
    so on), so ``prob_f`` and ``prob_d`` are fixed over time;
  * the elasticity of ``f`` is 0.0 — there is no lock-in for forest;
  * the elasticity of ``d`` is 0.6, but ``d`` is irreversible under the
    transition matrix, so it never comes into play;
  * class ``o`` is static and demand is monotonic.

For every ``f`` cell the decision reduces to
``prob_d + iter_d > prob_f + iter_f`` — a pure threshold on a static quantity.
The CLUE-S iterations are merely a slow search for that threshold.

Usage
-----
    from benchmark.naive_baseline import naive_allocation
    pred = naive_allocation(gdf_input, n_target=442)

See ``tests/test_benchmark_discriminance.py`` for the test that compares this
baseline against the full CLUE-S run.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logistic_probability(gdf, spec) -> np.ndarray:
    """Logistic probability of a ``LogisticRegressionSpec``, ignoring elasticity."""
    z = np.full(len(gdf), float(spec.const))
    for column, beta in (spec.betas or {}).items():
        z += float(beta) * gdf[column].values.astype(float)
    return _sigmoid(z)


def naive_allocation(gdf, n_target: int, potential_data=None) -> np.ndarray:
    """Allocate deforestation by pure static ranking.

    Parameters
    ----------
    gdf
        Input GeoDataFrame, with columns ``f``, ``d``, ``o`` and the covariates
        used by the regressions.
    n_target
        Total number of ``d`` cells wanted in the final state.
    potential_data
        The scenario's ``POTENTIAL_DATA`` structure. Defaults to the one in
        ``lucc_validation_executor``.

    Returns
    -------
    np.ndarray
        Binary deforestation vector for the final state, aligned to the rows
        of ``gdf``.
    """
    if potential_data is None:
        from disslucc_discrete.executors.lucc_validation_executor import POTENTIAL_DATA
        potential_data = POTENTIAL_DATA

    specs = potential_data[0]
    prob_f = logistic_probability(gdf, specs[0])
    prob_d = logistic_probability(gdf, specs[1])
    margin = prob_d - prob_f          # STATIC quantity — does not change over time

    d0 = (gdf["d"].values >= 0.5).astype(int)
    o0 = (gdf["o"].values >= 0.5).astype(int)
    eligible = (d0 == 0) & (o0 == 0)

    n_new = int(n_target) - int(d0.sum())
    if n_new <= 0:
        return d0

    order = np.argsort(-margin)
    order = order[eligible[order]]
    selected = order[:n_new]

    result = d0.copy()
    result[selected] = 1
    return result
