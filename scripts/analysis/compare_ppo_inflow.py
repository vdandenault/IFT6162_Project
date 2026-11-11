"""Train five past-inflow PPO seeds and compare with verified four-feature controls."""
from reservoir_control.systems.lake_como.ppo_inflow_study import main

if __name__ == "__main__":
    main()
