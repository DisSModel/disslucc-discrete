"""
disslucc.demand.precomputed
--------------------------
Pre-computed per-step demand — substrate-neutral (no GDF dependency).
Compatible with DemandProtocol; works with both vector and raster substrates.
"""
from __future__ import annotations

import csv
import io

from dissmodel.core import Model


def load_demand_csv(
    raw: str,
    land_use_types: list[str],
) -> list[list[float]]:
    """
    Parse a demand CSV string into a list ready for DemandPreComputedValues.

    Expected format — one column per land use type, one row per step:

        f,d,outros
        137878.17,19982.63,6489.20
        137622.22,20238.58,6489.20
        ...

    Column order does not need to match land_use_types; mapping is done
    by header name.

    This function is pure parsing — no I/O. The caller is responsible for
    reading the raw string from any source (local path, s3://, http://):

        from dissmodel.io._utils import read_text
        raw = read_text("s3://bucket/demand.csv")
        demand = load_demand_csv(raw, ["f", "d", "outros"])

    Parameters
    ----------
    raw : str
        CSV content as a string.
    land_use_types : list[str]
        Land use names in the order the demand values should be returned.

    Returns
    -------
    list[list[float]]
        [step][land_use] in land_use_types order.

    Raises
    ------
    ValueError
        If any land use type is missing from the CSV header.
    """
    reader  = csv.DictReader(io.StringIO(raw))
    missing = [lu for lu in land_use_types if lu not in reader.fieldnames]
    if missing:
        raise ValueError(
            f"Columns missing from demand CSV: {missing}\n"
            f"Available columns: {list(reader.fieldnames)}"
        )
    return [
        [float(row[lu]) for lu in land_use_types]
        for row in reader
    ]


class DemandPreComputedValues(Model):
    """
    Pre-computed per-step land use demand.

    Parameters
    ----------
    annual_demand : list[list[float]]
        [step][land_use] — index 0 is the initial step (env.now() == 0).
        Use load_demand_csv() to build this from a CSV string.
    land_use_types : list[str]
        Land use names in the same column order as annual_demand.

    Example
    -------
    from dissmodel.io._utils import read_text
    from disslucc import load_demand_csv, DemandPreComputedValues

    raw    = read_text("s3://bucket/demand.csv")   # caller resolves URI
    demand = DemandPreComputedValues(
        annual_demand  = load_demand_csv(raw, ["f", "d", "outros"]),
        land_use_types = ["f", "d", "outros"],
    )
    """

    INCREASING, DECREASING, STATIC = 1, -1, 0

    def setup(
        self,
        annual_demand:  list[list[float]],
        land_use_types: list[str],
    ) -> None:
        self.annual_demand    = annual_demand
        self.land_use_types   = land_use_types
        self.num_lu           = len(land_use_types)
        self.current_demand   = annual_demand[0]
        self.previous_demand  = annual_demand[0]
        self.demand_direction = [self.STATIC] * self.num_lu

    def execute(self) -> None:
        step = int(self.env.now())
        self.current_demand  = self.annual_demand[step]
        self.previous_demand = self.annual_demand[step - 1] if step > 0 else self.annual_demand[0]
        for i in range(self.num_lu):
            prev, curr = self.previous_demand[i], self.current_demand[i]
            if   step == 0 or prev == curr: self.demand_direction[i] = self.STATIC
            elif prev < curr:               self.demand_direction[i] = self.INCREASING
            else:                           self.demand_direction[i] = self.DECREASING

    # ── DemandProtocol ────────────────────────────────────────────────────────

    def get_current_lu_demand(self, i: int) -> float:  return self.current_demand[i]
    def get_previous_lu_demand(self, i: int) -> float: return self.previous_demand[i]
    def get_current_lu_direction(self, i: int) -> int: return self.demand_direction[i]

    def change_lu_direction(self, i: int) -> int:
        self.demand_direction[i] *= -1
        return self.demand_direction[i]