import numpy as np

from reservoir_control.systems.loslr.operator import ONTARIO_COST_COMPONENTS, ontario_stage_costs
from reservoir_control.systems.loslr.physics import CONVERSION, FLOW_SCALE


def advance_ontario(env, mdp, forcing, committed, combined, flow_tolerance):
    """Apply the accepted Ontario release and check its lake and reach balances."""
    k = mdp.t
    level = float(mdp.state["ontLevelBOQ"][k])
    flow = committed / FLOW_SCALE
    mdp.params.update(forcing)
    mdp.update(actions={"ont": {"flow": np.asarray([flow], dtype=np.float64)}})
    mdp = env.simulate(mdp)
    applied = float(mdp.state["ontFlow"][k + 1]) * FLOW_SCALE
    routed = float(mdp.state["stlouisFlow"][k + 1]) * FLOW_SCALE
    end_level = float(mdp.state["ontLevelBOQ"][k + 1])
    level_residual = end_level - level - (forcing["ontNTS"] - applied) / (FLOW_SCALE * CONVERSION)
    if (
        applied != committed
        or abs(routed - combined) > flow_tolerance
        or abs(level_residual) > 0.500001e-6
    ):
        raise ArithmeticError(f"Ontario flow, routing or level mismatch at period {k + 1}")
    return mdp, applied, routed, end_level, level_residual


def score_ontario(mdp, forcing, flow, start_level, end_level):
    period = mdp.t
    cost_inputs = {
        **forcing,
        **{name: series[period] for name, series in mdp.state.data.items()},
        "ontFlow": flow,
        "ontLevel": (start_level + end_level) / 2,
    }
    costs = ontario_stage_costs({name: [value] for name, value in cost_inputs.items()})[0]
    if not np.isfinite(costs).all():
        raise ArithmeticError(f"Nonfinite executed Ontario costs at period {period}")
    return dict(zip(ONTARIO_COST_COMPONENTS, costs, strict=True))
