# Ontario–Ottawa project handoff

1. **Baseline and simulator.** Reproduce the pinned Lake Ontario–St. Lawrence
   release and routing code. Keep the Plan 2014 rule curve as the operator
   baseline. Ontario forcing is historical. Ottawa storage, inflow scenarios
   and routing are explicit aggregate assumptions.
2. **Controllers.** Compare the rule curve, external-ranked MPC, Stackelberg
   MPC and five PPO seeds through a shared execution interface. The Ontario
   leader issues executable releases and the Ottawa follower solves a
   hard-constrained quadratic program.
3. **Evaluation.** Keep the 2017 and 2019 cases separate from the 18 other
   annual replays. Report completion, realized violations, operating cost
   and flood exposure for both lakes. Validate downstream stage separately
   against official gauge observations.
4. **Handoff.** Keep the selected controller comparison frozen. Reopen
   release-dependent routing and headroom validation when matched Carillon
   and outlet flows or a validated hydraulic model become available.

## Architecture

- `src/bilevel_reservoirs` contains the JAX Ontario transition, operator
  adapters and coupled evaluation entry points.
- `src/reservoir_control/controllers` contains the shared MPC and PPO
  algorithms. They operate through the Adapter pattern and do not import
  the reservoir systems.
- `src/reservoir_control/systems` contains Ontario and Ottawa physics,
  hard follower constraints, routing and field-data validation.
- `configs/studies` declares the model and split assumptions separately
  from the controller implementations.

To replace the research reference, require lower total Lake St. Louis
exposure, zero added Ontario shoreline exposure in each matched year,
preserved completion and no added physical violations.
[Results](RESULTS.md) · [Run instructions](README.md)
