import pytest
from dissmodel.core import Environment
from disslucc_discrete.components.demand.precomputed import DemandPreComputedValues

def test_demand_direction_logic():
    # Demand increasing for 'f', decreasing for 'd', static for 'o'
    annual_demand = [
        [100, 50, 10], # Step 0
        [110, 40, 10], # Step 1
    ]
    lu_types = ["f", "d", "o"]
    
    # Create a real environment as required by the Model base class
    env = Environment(start_time=0, end_time=2)
    
    demand = DemandPreComputedValues(
        annual_demand=annual_demand, 
        land_use_types=lu_types
    )
    
    # Step 0
    # env._now is set directly here because Environment has no public advance() method —
    # the only public API is run(), which executes the full simulation. This is deliberate:
    # the test isolates DemandPreComputedValues.execute() without running the full lifecycle.
    # Candidate for a dissmodel issue: expose env.advance(n) to step the clock by n ticks.
    env._now = 0
    demand.execute()
    assert demand.get_current_lu_direction(0) == 0 # Static at start
    assert demand.get_current_lu_demand(0) == 100
    
    # Step 1 — same reasoning as above: direct clock manipulation to isolate the unit.
    env._now = 1
    demand.execute()
    assert demand.get_current_lu_direction(0) == 1  # 'f' increased (100 -> 110)
    assert demand.get_current_lu_direction(1) == -1 # 'd' decreased (50 -> 40)
    assert demand.get_current_lu_direction(2) == 0  # 'o' static (10 -> 10)
    assert demand.get_current_lu_demand(0) == 110
