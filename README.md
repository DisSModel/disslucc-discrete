# DisSLUCC-Discrete 🌍

> **Discrete Spatial Library for Land Use Change Modeling** — A Python implementation of discrete LUCC modeling components (CLUE-S like), built on top of [DisSModel](https://github.com/DisSModel/dissmodel)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![LambdaGeo](https://img.shields.io/badge/LambdaGeo-Research-green.svg)](https://github.com/DisSModel)

---

## 📖 About

**DisSLUCC-Discrete** is a Python library that implements spatially explicit components for discrete Land Use and Cover Change (LUCC) modeling. It provides a CLUE-S like allocation algorithm with logistic regression for potential estimation.

This package focuses on **discrete** land use change (one use per cell), following the philosophy described by **Verburg et al. (2002)**.

> ℹ️ **Note**: "DisSModel" is spelled with a capital S in the middle, standing for **S**patial.

---

## 🚀 Quick Start

### Running a simulation (Moju example)

You can run a discrete LUCC simulation using the `ClueSVectorExecutor`. You will need a TOML configuration file (for regression coefficients and rules) and a CSV file (for demand values).

```bash
# Run the simulation using the CLI executor
python src/disslucc_discrete/executors/clue_s_vector_executor.py run \
  --input data/cs_moju.zip \
  --output outputs/resultado_moju.gpkg \
  --toml examples/moju_model.toml \
  --param demand_csv=examples/data/demand_moju.csv \
  --param n_steps=6
```

### Configuration (model.toml)

The model behavior is defined in a TOML file. It includes the land use types, logistic regression coefficients (betas), elasticities, and the transition matrix.

```toml
[model]
land_use_types = ["f", "d", "o"]
region_attr    = "region"

[[model.potential]]
# Coefficients for land use "f"
const      = -2.3418
elasticity = 0.0
[model.potential.betas]
media_decl = -0.0272
dist_br    = 3.1031

[[model.potential]]
# Coefficients for land use "d"
const      = -0.1003
elasticity = 0.6
[model.potential.betas]
dist_br    = -2.5165

[model.allocation]
max_difference   = 10.0
max_iteration    = 1000
factor_iteration = 0.0001

[model.transition_matrix]
data = [[[1, 1, 0], [0, 1, 0], [0, 0, 1]]]
```

### Demand (demand.csv)

The demand CSV should contain one column per land use type and one row per time step.

```csv
f,d,o
5706,205,3
5658,253,3
5611,300,3
...
```

### Running the Lab6 validation (TerraME parity check)

To reproduce the 100% cell-level parity result against TerraME/LuccME:

```bash
python src/disslucc_discrete/executors/lucc_validation_executor.py run \
  --input  data/cs_moju.zip \
  --output outputs/validation \
  --param  terrame_reference=benchmark/data/Lab6_2004.zip
```

Artifacts (report.md, scatter.png, map.png, lab6_python_2004.zip) are written
to `outputs/validation/`.

### Using the Makefile facilitator

You can also use the following shorthand:

```bash
# Run Moju simulation
make run-moju

# Format code
make format

# Run linting
make lint
```

---

## 🧪 Testing & Validation

The primary validation strategy is cell-by-cell parity against the TerraME/LuccME
reference implementation (Lab6, Moju dataset, 1999–2004). This test is automated
and runs on every CI build.

```bash
pytest tests/ -v
```

The integration test in `tests/test_validation_lab6.py` instantiates
`LuccValidationExecutor`, runs the simulation over `data/cs_moju.zip`, and asserts:

| Metric | Expected |
|---|---|
| **Accuracy** | 100.00% |
| **Quantity disagreement** (Pontius & Millones, 2011) | 0.000000 |
| **Allocation disagreement** (Pontius & Millones, 2011) | 0.000000 |
| **F1 Score** | 1.0000 |
| **FP / FN** | 0 / 0 |

Cohen's κ is still computed for backward compatibility with older reports, but it
is **deprecated** across the DisSModel ecosystem in favour of the Pontius &
Millones quantity/allocation decomposition, and is no longer asserted.

To run only the integration test:

```bash
pytest tests/test_validation_lab6.py -v
```

---

## 📊 Validation

The discrete implementation has been validated against the original **TerraME/LuccME (Lab6)** reference using the Moju dataset (1999–2004). The Python implementation achieves **100% numerical parity** at the cell level.

| Metric | Value |
|---|---|
| **Accuracy** | 100.00% |
| **Quantity disagreement** | 0.000000 |
| **Allocation disagreement** | 0.000000 |
| **F1 Score** | 1.0000 |
| **Runtime** | ~65 ms/step |

> ### ⚠️ Discriminance caveat — read before citing this result
>
> The parity above is real and reproducible, but the Lab6 scenario is close to
> **non-discriminative**. A trivial static ranking by `prob_d - prob_f` — with no
> CLUE-S, no iteration, no time steps and no DisSModel at all — reproduces the same
> TerraME output cell for cell (5914/5914).
>
> This happens because every covariate in Lab6 is static, the elasticity of `f` is
> `0.0`, and `d` is irreversible, so the allocation collapses to a pure threshold on
> a fixed quantity. The CLUE-S iterations merely search for that threshold.
>
> **What this benchmark actually validates:** that the logistic regression
> coefficients were transcribed correctly from the Lua original. It does **not**
> exercise the allocation loop, the transition matrix beyond irreversibility,
> regional stratification, `tau`, or multi-step dynamics.
>
> See `benchmark/naive_baseline.py` and `tests/test_benchmark_discriminance.py`.
> Please do not cite this result as evidence of CLUE-S algorithmic fidelity.

To run the parity benchmark:

```bash
make benchmark
```

Results (maps, scatter plots, and reports) are generated in `benchmark/results/`.

---

## 🧩 Core Components

DisSLUCC-Discrete implements the three-pillar LUCC modeling philosophy:

### 1️⃣ Demand Component

Computes the magnitude of land-use change to allocate at each time step.

- `DemandPreComputedValues`: Handles pre-calculated demand values from CSV.

### 2️⃣ Potential Component

Estimates the suitability of each cell to change.

- `PotentialDLogisticRegression`: Implements logistic regression for discrete suitability.

### 3️⃣ Allocation Component

Spatially distributes changes.

- `AllocationDClueSLike`: Discrete competition-based allocation (CLUE-S).

---

## 🗂️ Executor Architecture

DisSLUCC follows the DisSModel `ModelExecutor` pattern.

- `ClueSVectorExecutor`: Executor for discrete simulations on vector substrates (GeoDataFrame).

---

## 📦 Installation

```bash
cd disslucc-discrete
pip install -e .
```

**Dependencies:** `dissmodel`, `geopandas`, `pandas`, `numpy`, `rasterio`

---

## 📚 References

- **CLUE-S**: Verburg, P. H., Soepboer, W., Veldkamp, A., Limpiada, R., Espaldon, V., & Mastura, S. S. (2002). Modeling the spatial dynamics of regional land use: the CLUE-S model. *Environmental management*, 30(3), 391-405.

---

## 📄 License

Distributed under the **MIT License**. Developed by the **[LambdaGeo](https://lambdageo.github.io)** research group.