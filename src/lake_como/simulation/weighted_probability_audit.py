"""Reproduce probability diagnostics before/after five-path reduction.

Supplemental checks only: never used to revise the frozen forecast selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.environment import settings
from .weighted_study import DEFAULT,FOLDS,LATE
from .weighted_forecast import WeightedForecastBank,mixture_distribution,CONFIGS,validate_probabilities,weighted_crps
from .reserve_use_check import save,verify


def distribution_metrics(values,probabilities,truth,wet_threshold):
    x=np.asarray(values,float);p=validate_probabilities(probabilities,len(x))
    if not np.isfinite(x).all() or not np.isfinite([truth,wet_threshold]).all():raise ValueError('Finite scores required')
    order=np.argsort(x,kind='stable');x=x[order];p=p[order];cdf=np.cumsum(p);cdf[-1]=1.
    # Half of the weighted pairwise absolute-distance sum, computed in O(N).
    crps=float(np.sum(p*abs(x-truth))-np.sum(p*x*(2*cdf-p-1)))
    prob=float(p[x>wet_threshold].sum());wet=truth>wet_threshold
    row=dict(volume_crps=crps,wet_truth=wet,wet_probability=prob,brier=(prob-float(wet))**2)
    for name,level in [('q50',.5),('q90',.9),('q95',.95)]:
        row[name+'_covered']=bool(truth<=x[np.flatnonzero(cdf>=level-1e-12)[0]])
    return row


def run(root):
    verify(root);out=root/'probability_audit';out.mkdir(exist_ok=False)
    selected=json.loads((root/'selection.json').read_text())['selected_variant']
    files=[Path(__file__),root/'plan.json',root/'selection.json',root/'late_check_plan.json',root/'late_check.json']
    for fold in [*FOLDS,LATE]:files.extend([root/(fold['name']+'.csv'),root/(fold['name']+'_record.json')])
    save(out/'plan.json',dict(source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        representation='Five weighted control representatives versus full underlying finite mixture. Discrete quantiles use a 1e-12 CDF arithmetic tolerance.',
        used_for_selection=False,new_controller_runs=False,test_evaluated=False))
    training=settings('train');rows=[]
    for fold in [*FOLDS,LATE]:
        threshold=json.loads((root/(fold['name']+'_record.json')).read_text())['wet_threshold_million_m3']
        scored=pd.read_csv(root/(fold['name']+'.csv'))
        for variant in list(dict.fromkeys(['legacy',selected])):
            bank=WeightedForecastBank(training['inflow'],training['demand'],before_year=fold['before_year'],variant=variant)
            alpha=0. if variant=='legacy' else CONFIGS[variant]['alpha']
            original=scored[scored.variant.eq(variant)].set_index(['year','model_day','seed'])
            for year in fold['years']:
                q=np.asarray(training['inflow'][year][:365],float)
                for n in range(7,359):
                    past=q[n-7:n].astype(np.float32).astype(float);bank.set_history(past)
                    truth=float(q[n:n+7].sum()*86400/1e6)
                    paths,p,_=mixture_distribution(bank,n+1,past,alpha)
                    rows.append(dict(fold=fold['name'],variant=variant,representation='full_mixture',year=year,model_day=n+1,seed=None,
                        **distribution_metrics(paths.sum(axis=1)*86400/1e6,p,truth,threshold)))
                    for seed in [901,902,903,904,905]:
                        paths,_,meta=bank.forecast('conditional',n+1,float(past[-1]),seed=seed+n*24)
                        metrics=distribution_metrics(paths.sum(axis=1)*86400/1e6,meta['scenario_probabilities'],truth,threshold)
                        np.testing.assert_allclose(metrics['volume_crps'],original.loc[(year,n+1,seed),'volume_crps'],atol=1e-9,rtol=0)
                        rows.append(dict(fold=fold['name'],variant=variant,representation='five_paths',year=year,model_day=n+1,seed=seed,**metrics))
            print('Probability diagnostics',fold['name'],variant,'done',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'probabilities.csv',index=False);summaries=[]
    for scope,g in [('development',f[f.fold.isin([x['name'] for x in FOLDS])]),('late_training_check',f[f.fold.eq(LATE['name'])])]:
        for (variant,representation),data in g.groupby(['variant','representation']):
            summaries.append(dict(scope=scope,variant=variant,representation=representation,observations=len(data),
                **{name:float(data[name].mean()) for name in ['volume_crps','brier','wet_probability','wet_truth','q50_covered','q90_covered','q95_covered']}))
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    save(out/'run.json',dict(complete=True,selection_unchanged=True,used_for_selection=False,new_controller_runs=False,test_evaluated=False))
    verify(out);verify(root);print(pd.DataFrame(summaries).to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=DEFAULT)
    run(p.parse_args().study)


if __name__=='__main__':main()
