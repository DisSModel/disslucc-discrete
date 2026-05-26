# DisSLUCC-Discrete 🌍

> **Discrete Spatial Library for Land Use Change Modeling** — A Python implementation of discrete LUCC modeling components (CLUE-S like), built on top of [DissModel](https://github.com/LambdaGeo/dissmodel)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![LambdaGeo](https://img.shields.io/badge/LambdaGeo-Research-green.svg)](https://github.com/LambdaGeo)

---

## 📖 About

**DisSLUCC-Discrete** is a Python library that implements spatially explicit components for discrete Land Use and Cover Change (LUCC) modeling. It provides a CLUE-S like allocation algorithm with logistic regression for potential estimation.

This package focuses on **discrete** land use change (one use per cell), following the philosophy described by **Verburg et al. (2002)**.

---

## 🚀 Quick Start

### Running a simulation (Moju example)

You can run a discrete LUCC simulation using the `ClueSVectorExecutor`. You will need a TOML configuration file (for regression coefficients and rules) and a CSV file (for demand values).

```bash
# Run the simulation using the CLI executor
python src/disslucc/executors/clue_s_vector_executor.py run \
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

DisSLUCC follows the DissModel `ModelExecutor` pattern.

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

Distributed under the **MIT License**. Developed by the **[LambdaGeo](https://github.com/LambdaGeo)** research group.
