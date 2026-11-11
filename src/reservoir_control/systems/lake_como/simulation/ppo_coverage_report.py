"""Derived ledgers and common-stratum diagnostics, never a new model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from reservoir_control.systems.lake_como.simulation.ppo_coverage import ROOT,SEEDS,ARMS,IDENTIFICATION,dataset,save,verify


def report(root):
    verify(json.loads((root/'plan.json').read_text()));verify(json.loads((root/'dataset_manifest.json').read_text()))
    annual=pd.read_csv(root/'seed_factual_annual.csv',float_precision='round_trip')
    ledgers=[]
    for (seed,block),group in annual.groupby(['seed','block_start']):
        before=group[group.arm.eq('baseline')];after=group[group.arm.eq('treatment')]
        if len(before)!=2 or len(after)!=2 or not group.complete.all():raise ValueError('Complete paired block required')
        delivered=float(after.delivered_target_million_m3.sum()-before.delivered_target_million_m3.sum())
        river=float(after.remaining_river_million_m3.sum()-before.remaining_river_million_m3.sum())
        storage=float(after.loc[after.position_in_pair.eq(2),'end_storage_million_m3'].iloc[0]-before.loc[before.position_in_pair.eq(2),'end_storage_million_m3'].iloc[0])
        unmet=float(after.unmet_target_million_m3.sum()-before.unmet_target_million_m3.sum())
        if abs(delivered+river+storage)>1e-7 or abs(delivered+unmet)>1e-7:raise ValueError('Paired water allocation ledger fails')
        ledgers.append(dict(seed=int(seed),block_start=int(block),delta_delivered_million_m3=delivered,delta_remaining_river_million_m3=river,
            delta_episode_end_stock_million_m3=storage,ledger_error_million_m3=delivered+river+storage))
    pd.DataFrame(ledgers).to_csv(root/'seed_stock_ledgers.csv',index=False)
    actions=pd.read_csv(root/'seed_common_actions.csv',float_precision='round_trip')
    actions['level_group']=pd.cut(actions.level,[-np.inf,-.45,.35,1.24,np.inf],labels=['low','lower','retained','flood'])
    actions.groupby(['seed','level_group'],observed=True).agg(states=('ordinal','size'),release=('first_release','mean'),branch_cost=('branch_cost','mean'),end_level=('branch_end_level','mean')).to_csv(root/'seed_common_level_groups.csv')
    target=actions[actions.seed.eq(124)].set_index(['block_start','ordinal'])
    rest=actions[~actions.seed.eq(124)].groupby(['block_start','ordinal']).mean(numeric_only=True).reindex(target.index)
    contrast=pd.DataFrame({field:target[field]-rest[field] for field in ['request','first_release','branch_cost','branch_end_level','predicted_storage_slope']})
    contrast['level']=target.level;contrast.to_csv(root/'seed124_common_contrasts.csv')
    error_groups=[]
    for seed in SEEDS:
        _,truth=dataset(IDENTIFICATION,seed,'check');arrays={}
        for arm in ARMS:
            with np.load(root/f'seed{seed}/{arm}_checking.npz') as data:arrays[arm]={key:data[key].copy() for key in ['error','nearest']}
        # Common strata are defined from concentrated-state distances, not each
        # treatment's own error ordering. This is post hoc descriptive analysis.
        boundaries=np.quantile(arrays['concentrated']['nearest'],[.5,.9])
        strata=np.digitize(arrays['concentrated']['nearest'],boundaries,right=True)
        for arm in ARMS:
            for index,label in enumerate(['nearest_half','middle_40pct','furthest_10pct']):
                mask=strata==index;error=arrays[arm]['error'][mask]
                error_groups.append(dict(seed=seed,arm=arm,support_group=label,count=int(mask.sum()),mse=float(np.mean(error**2)),mae=float(abs(error).mean()),bias=float(error.mean())))
    pd.DataFrame(error_groups).to_csv(root/'coverage_support_strata.csv',index=False)
    summary=dict(seed124_ledger_mean={key:float(value) for key,value in pd.DataFrame(ledgers).query('seed==124').drop(columns=['seed','block_start']).mean().items()},
        seed124_common_mean_contrasts={key:float(value) for key,value in contrast.drop(columns='level').mean().items()},
        common_level_state_counts=actions[actions.seed.eq(124)].level_group.value_counts().to_dict(),
        interpretation='All derived analyses are post hoc descriptions. Seed124 is retained alongside all five seeds; remaining river flow is not measured waste. Thirty-day branches do not price terminal stock.',test_evaluated=False)
    save(root/'derived_summary.json',summary)
    save(root/'derived_sources.json',dict(source_sha256={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args();report(args.root)
