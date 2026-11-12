---
created: 2026-11-11 17:14:46
file_class:
modified: 2026-11-11T19:19:00-05:00
---
![[Shortcuts]]

---

## Step 1: Write an easy-to-use python interface for the Lake Ontario+St. Lawrence pipeline

Basically something that lets us do this:

```python
next_state, outputs = simulator.step(
    state=current_state,
    releases=executed_releases,
    external_inputs=period_inputs,
)
```

The **inputs** would be:

| Input                          | What it contains                                                                                                                                                       |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Current state**              | Lake Ontario’s level at the beginning of the period, the current quarter-month, and necessary history such as previous releases.                                       |
| **Executed releases (action)** | The Lake Ontario outflow to apply during this period, in m³/s.                                                                                                         |
| **External inputs**            | Lake Ontario’s net water supply, the historical combined contribution from Ottawa and other tributaries, ice conditions, and other required downstream routing inputs. |
And the outputs would be:

| Output           | What it contains                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`next_state`** | Lake Ontario’s updated level, the next quarter-month, and updated history such as the executed release.                                                             |
| **`outputs`**    | Lake Ontario’s mean and end-of-period levels; the calculated Lake St. Louis level and flow; other modeled St. Lawrence water levels; and water-balance diagnostics. |
## Step 2
### Step 2A: Write an MPC controller for the St. Lawrence flow

(completely independent from the Ottawa river stuff)

### Step 2B
#### Step 2B (1): Write a model / simulator of the Ottawa river**

Something like:

```python
next_ottawa_state, ottawa_outputs = ottawa_simulator.step(
    state=current_ottawa_state,
    releases=executed_ottawa_releases,
    external_inputs=ottawa_period_inputs,
)
```

**Input:** The controller takes the current Ottawa storage estimate, release history, flow forecasts, the announced Lake Ontario release plan, etc.

**Output:** It returns an Ottawa release plan, predicted storage and downstream flows and levels at lake St. Louis, etc.
#### Step 2B (2): Write an MPC controller

Build a controller for the Ottawa River. It should take the announced Lake Ontario release plan as an input (I guess this could be a "fake" plan while waiting for the St. Lawrence MPC controller)

Something like:

```python
ottawa_plan = ottawa_follower.respond(
    information=ottawa_information,
    lake_ontario_plan=proposed_release_sequence,
    other_flow_forecasts=other_flow_forecasts,
)
```

## Step 3: Combine both controllers with an ordinary MPC (not bilevel)

Run both controllers against one coupled physical simulator, including downstream routing + Châteauguay flow.

Lake Ontario MPC chooses its plan using a fixed Ottawa forecast. The Ottawa follower then responds to the announced plan.

## Step 4: Bilevel MPC/Stackelberg

Make the Lake Ontario optimizer anticipate how the Ottawa follower responds to each candidate Lake Ontario plan. Basically, while choosing its plan, Lake Ontario MPC evaluates how Ottawa would respond to each candidate plan.

## Step 5: Compare both results.

Compare ordinary and bilevel MPC using the same physical simulator, Ottawa follower, available information, forecasts, starting conditions, etc.

# Execution order

```
Look at available information and forecasts
                    ↓
Select a Lake Ontario release plan:
  • Ordinary MPC: hold the Ottawa-flow forecast fixed
  • Bilevel MPC: evaluate predicted Ottawa responses
    while searching over candidate Lake Ontario plans
                    ↓
Announce the selected Lake Ontario plan
                    ↓
Ottawa follower computes its response to the lake ontario plan
                    ↓
Execute the first release from each plan together
                    ↓
Advance the simulator using the realized inputs
                    ↓
Return updated states and outputs; repeat at the next quarter-month
```

