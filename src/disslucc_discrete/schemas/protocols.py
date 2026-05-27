"""
disslucc.schemas.protocols
--------------------------
Protocolos que definem as interfaces entre os componentes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DemandProtocol(Protocol):
    def get_current_lu_demand(self, lu_index: int) -> float: ...
    def get_previous_lu_demand(self, lu_index: int) -> float: ...
    def get_current_lu_direction(self, lu_index: int) -> int: ...
    def change_lu_direction(self, lu_index: int) -> int: ...


@runtime_checkable
class PotentialProtocol(Protocol):
    def modify(self, r_number: int, lu_idx: int, direction: int) -> None: ...
