import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from lake_como.environment import settings
from lake_como.simulation import make_operational_env,PhysicalWeights,Operations
from lake_como.simulation.controllers import ContinuousMPC,ContinuousResidual,ScenarioPlanner,ScenarioConstraintFilter
from lake_como.simulation.scenarios import HistoricalScenarios
from lake_como.simulation.validation import measured_balance

class OperationalControlChecks(unittest.TestCase):
    def test_mpc_horizon_and_warmstart_cross_year_without_reset(self):
        env=make_operational_env();obs,info=env.reset()
        for _ in range(364):obs,_,_,_,info=env.step(np.array([100],np.float32))
        base=ContinuousMPC(settings('train'));base.reset()
        def solver(fun,initial,**kwargs):return SimpleNamespace(x=initial,success=True,status=0)
        with patch('lake_como.simulation.controllers.minimize',solver):
            action=base(obs,info)
            self.assertEqual(base.diagnostics[-1]['days'],[365,1,2,3,4,5,6])
            obs,_,_,_,info=env.step(action);base(obs,info)
        self.assertEqual(base.resets,1);self.assertTrue(base.diagnostics[-1]['used_shifted_plan'])
        env.close()

    def test_planner_and_filter_bounds_and_recovery(self):
        params=settings('train');bank=HistoricalScenarios(params['inflow'],params['demand'])
        env=make_operational_env(decision_hours=6,operations=Operations(ramp_m3_s_per_hour=5,river_residence_hours=6))
        obs,info=env.reset();planner=ScenarioPlanner(env.model,bank,PhysicalWeights(),horizon_days=1,scenarios=2,grid_points=3)
        request=planner(obs,info);self.assertTrue(env.action_space.contains(request))
        filt=ScenarioConstraintFilter(env.model,bank,PhysicalWeights(),scenarios=2,grid_points=3)
        high=obs.copy();high[0]=2.
        corrected=filt(request,high,info)
        self.assertTrue(env.action_space.contains(corrected));self.assertTrue(filt.diagnostics[-1]['recovery'])
        self.assertIn('no unconditional',filt.diagnostics[-1]['guarantee'])

    def test_residual_adapter_uses_carried_inflow_without_january_reset(self):
        class Base:
            def __init__(self):self.resets=0
            def reset(self):self.resets+=1
            def __call__(self,obs,info):return np.array([100],np.float32)
        class Agent:
            mode='residual';high=491.61
            def __init__(self):self.seen=[]
            def act(self,obs,deterministic):self.seen.append(obs.copy());return np.zeros(1,np.float32)
            def physical_action(self,value):return value*50
        agent=Agent();base=Base();controller=ContinuousResidual(agent,base);controller.reset()
        obs=np.array([.35,0,1,90,123,100,0,.5],np.float32)
        controller(obs,{});obs[4]=456;controller(obs,{})
        self.assertEqual(base.resets,1)
        self.assertEqual(agent.seen[0][4],123);self.assertEqual(agent.seen[1][4],456)
        self.assertEqual(agent.seen[1][5],100)

    def test_planner_includes_declared_finite_terminal_reserve(self):
        class Bank:
            demand_climatology=np.zeros(365)
            def paths(self,day,last_inflow,*,horizon,count,seed):
                return np.full((count,horizon),100.),{}
        weights=PhysicalWeights(shortage_per_million_m3=0,flood_per_m_hour=0,
            low_water_per_m_hour=0,downstream_per_million_m3=0,
            terminal_reserve_per_million_m3=1000,reserve_target_level_m=1.25)
        env=make_operational_env(years=(1946,),weights=weights)
        obs,info=env.reset();info.update(elapsed_hours=364*24,remaining_hours=24,next_day=365)
        planner=ScenarioPlanner(env.model,Bank(),weights,horizon_days=1,scenarios=1,grid_points=2)
        planner(obs,info)
        self.assertGreater(planner.diagnostics[-1]['expected_physical_cost'],0)
        env.close()

    def test_measured_balance_requires_explicit_alignment(self):
        frame=pd.DataFrame(dict(timestamp=['2001-01-01','2001-01-02'],level_m=[.35,.35],inflow_m3_s=[100,100],outflow_m3_s=[100,100]))
        with self.assertRaises(ValueError):measured_balance(frame,{})
        meta=dict(source_url='synthetic-test',timezone='UTC',level_datum='test datum',alignment_confirmed=True,
            flow_convention='interval_mean_until_next_timestamp',level_convention='at_timestamp',units={'level':'m','inflow':'m3/s','outflow':'m3/s'})
        self.assertEqual(measured_balance(frame,meta)['balance_rmse_m3'],0)
        meta['alignment_confirmed']=False
        with self.assertRaises(ValueError):measured_balance(frame,meta)
