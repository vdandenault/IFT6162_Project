"""Finite-candidate, perfect-information coupling smoke test (not historical operations)."""
import argparse
import ast
from contextlib import chdir
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from reservoir_control.systems.ottawa.hard_follower_study import (
    CONFIG, ROOT, FollowerTolerances, TransitState, make_problem, solve_hard_follower,
)


def pinned_methods(path, digest, owner, names, namespace):
    """Load the indicated framework-independent methods of the archived source only."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise ValueError(f"Unreviewed simulator version: {path}")
    cls = next(node for node in ast.parse(source).body if getattr(node, "name", "") == owner)
    for name in names:
        method = next(node for node in cls.body if getattr(node, "name", "") == name)
        method.decorator_list = []
        exec(compile(ast.Module([method], []), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def load_physics(folder):
    folder = folder / "src/bilevel_reservoirs"
    legacy = ROOT / "external/loslr_regulation_optimization"
    sys.path.insert(0, str(legacy))
    from functions.routing.stlaw import stLawrenceRouting
    from functions.limits.Bv7 import planLimits
    with chdir(legacy):
        from objectiveFunctions.legacyPIs.functions import (
            upstreamCoastalImpacts, downstreamCoastalImpacts, hydropowerProduction,
            recreationalBoating,
        )
    lake = pinned_methods(folder / "simulators/loslr.py",
        "e82704dde209235e0b31206611342f9232c0afb3ad31dda89695bd1435ef7cda",
        "LoslrEnv", ["lostep", "total_flow", "lrstep"],
        {"np": np, "Any": object, "stLawrenceRouting": stLawrenceRouting})
    limit = pinned_methods(folder / "controllers/operator.py",
        "e925d14e45893c6d05c50a65b1f246fd7272b04a951ccfc562188be110c6ba01",
        "AdjustFlow", ["constrain"], {}).constrain
    return lake, lambda *args: limit(SimpleNamespace(constraint_fn=planLimits), *args), (
        (upstreamCoastalImpacts, "upstreamCoastal", 1),
        (downstreamCoastalImpacts, "downstreamCoastal", 1),
        (hydropowerProduction, "totalEnergyValue", -1),
        (recreationalBoating, "totalRecBoating", 1))


def evaluate(rows, schedule, arrivals, background, capacities, state, physics, limits):
    lake, constrain, indicators = physics
    level, previous_flow, previous_ice = state
    records = []
    for row, total, arrival, other, capacity in zip(rows, schedule, arrivals, background, capacities, strict=True):
        native = total / 10
        supply = row["ontNTS"]
        target = level + (supply - total) / 29700
        adjustment = native - lake.total_flow(level, target, 0.)
        end = lake.lostep(level, adjustment, supply)
        if abs(target - end) > 1e-8 or abs(lake.total_flow(level, end, adjustment) - native) > 1e-7:
            raise ArithmeticError("Ontario transition/total discharge disagreement")
        nonont = other + arrival
        forcing = dict(row, ontNTS_QM1=supply, stlouisontOut=nonont, slonFlow_QM1=nonont)
        allowed, regime = constrain(forcing, level, end, native, "RF", previous_flow, previous_ice)
        if abs(allowed - native) > 1e-7:
            return None, f"Bv7 {regime} changes commitment at QM {row['QM']}"
        routed = lake.lrstep(level, end, native, dict(forcing, slonFlow=nonont, jetty1R=row["varennesR"]))
        combined = 10 * routed["stlouisFlow"]
        if not np.isclose(combined, total + nonont, rtol=0, atol=1e-6):
            raise ArithmeticError("Routing boundary/unit disagreement")
        unrounded = 16.57 + (row["ptclaireR"] * combined / 604)**0.58
        if end > limits["ontario_cap_m"] or unrounded > limits["pointe_claire_cap_m"] + 1e-8 or combined > capacity + 1e-5:
            return None, "hard flood limit"
        records.append(forcing | routed | dict(ontLevel=(level+end)/2, ontFlow=native,
            ont_end_m=end, ont_release_m3_s=total, arrival_m3_s=arrival, combined_m3_s=combined,
            cap_m3_s=capacity, pointe_unrounded_m=unrounded))
        level, previous_flow, previous_ice = end, native, row["iceInd"]
    data = pd.DataFrame(records).to_dict("list")
    cost = sum(sign * model.runModel(data)[key].to_numpy() for model, key, sign in indicators)
    if not np.isfinite(cost).all():
        raise ArithmeticError("Nonfinite Ontario cost")
    for record, stage_cost in zip(records, cost, strict=True):
        record["stage_cost"] = float(stage_cost)
    return records, "ok"


def run(args):
    physics = load_physics(args.ontario)
    source = args.ontario / "data/input/historic/1961_2020/12month_sqAR.txt"
    frame = pd.read_table(source)
    rows = frame[(frame.Year == 1995) & frame.QM.between(17, 28)].to_dict("records")
    assert len(rows) == 12
    config = json.loads(CONFIG.read_text())
    scenario = next(s for s in config["scenarios"] if s["id"] == "one_period_delay")
    limits = json.loads((ROOT/"configs/studies/loslr/scoring.json").read_text())
    tol = FollowerTolerances(**config["acceptance"])
    dt = config["clock"]["duration_seconds"]
    log, trace, summary = [], [], {}
    for mode in ("anticipating", "external"):
        storage, previous = config["initial_storage_m3"], config["previous_executed_release_m3_s"]
        transit = TransitState(tuple(scenario["initial_transit_volumes_m3"]))
        state = (74.8, 700., 0.)
        for step in range(len(rows)):
            window = rows[step:step+5]
            horizon = len(window)
            base = make_problem(config, scenario, step, storage, previous, transit,
                scenario["realized_net_inflow_m3_s"][step:step+horizon])
            background = base.other_downstream_m3_s
            capacities = np.minimum(scenario["downstream_capacity_m3_s"][step:step+horizon],
                [604 / row["ptclaireR"] * (limits["pointe_claire_cap_m"]-16.57)**(1/0.58) for row in window])
            options = []
            for choice in (6400, 6700, 7000, 7300, 7600):
                schedule = np.full(horizon, float(choice))
                problem = replace(base, ontario_executable_release_m3_s=schedule,
                                  downstream_capacity_m3_s=capacities)
                entry = dict(mode=mode, period=step+1, candidate_m3_s=choice)
                log.append(entry)
                if mode == "anticipating":
                    response = solve_hard_follower(problem, tol)
                    entry["status"] = response.status
                    if not response.success:
                        if response.status != "infeasible": raise RuntimeError(response.message)
                        continue
                    arrivals = base.routing_matrix @ response.release_m3_s + base.in_transit_contribution_m3_s
                    entry["ottawa_first_m3_s"] = response.first_release_m3_s
                else:
                    response = None
                    arrivals = base.routing_matrix @ np.full(horizon, 900.) + base.in_transit_contribution_m3_s
                plan, entry["status"] = evaluate(window,schedule,arrivals,background,capacities,state,physics,limits)
                if plan is not None:
                    entry["score"] = sum(record["stage_cost"] for record in plan)
                    options.append((entry["score"], choice, problem, response))
            if not options:
                summary[mode] = f"no admissible candidate at period {step+1}"
                break
            _, choice, problem, response = min(options, key=lambda item: item[:2])
            if response is None:
                response = solve_hard_follower(problem, tol)
            if not response.success:
                if response.status != "infeasible": raise RuntimeError(response.message)
                summary[mode] = f"real follower infeasible at period {step+1}"
                break
            release = response.first_release_m3_s
            next_transit, arrival, error = transit.advance(release, dt)
            actual, reason = evaluate(window[:1],[choice],[arrival],background[:1],capacities[:1],state,physics,limits)
            if actual is None:
                summary[mode] = f"execution rejected at period {step+1}: {reason}"
                break
            record = actual[0]
            following = storage + dt*(base.net_inflow_m3_s[0]-release)
            if abs(following-response.storage_m3[1]) > tol.storage_m3 or abs(error)>0.01 or abs(record["combined_m3_s"]-response.downstream_flow_m3_s[0])>tol.flow_m3_s:
                raise ArithmeticError("Executed joint transition differs from follower prediction")
            trace.append(dict(mode=mode, period=step+1, ontario_m3_s=choice, ottawa_m3_s=release,
                storage_m3=following, transit_m3=next_transit.total_m3, **{k:record[k] for k in
                ("arrival_m3_s","combined_m3_s","cap_m3_s","ont_end_m","pointe_unrounded_m","stage_cost")}))
            state = (record["ont_end_m"], choice/10, record["iceInd"])
            storage, previous, transit = following, release, next_transit
        else:
            summary[mode] = "completed"
    args.output.mkdir(parents=True,exist_ok=False)
    executed = pd.DataFrame(trace)
    executed.to_csv(args.output/"executed.csv",index=False)
    pd.DataFrame(log).to_csv(args.output/"candidates.csv",index=False)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if not executed.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, layout="constrained")
        for axis, column, label in zip(axes, ["ontario_m3_s","ottawa_m3_s","combined_m3_s"],
                                       ["Ontario release","Ottawa release","Lake St. Louis flow"]):
            for mode, group in executed.groupby("mode"):
                axis.plot(group.period, group[column], marker=".", label=mode)
            axis.set_ylabel(label+"\n(m³/s)")
        caps = executed.drop_duplicates("period").sort_values("period")
        axes[-1].step(caps.period, caps.cap_m3_s, where="mid", linestyle="--", color="black", label="capacity")
        axes[0].legend(); axes[-1].legend(); axes[-1].set_xlabel("Quarter-month in experiment")
        fig.suptitle("Ontario–Ottawa coupling pilot · synthetic Ottawa · perfect information")
        fig.savefig(args.output/"comparison.png",dpi=180)
        plt.close(fig)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontario",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
