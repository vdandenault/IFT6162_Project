"""Reproduce the selected policy study with its mechanism plan frozen up front."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from reservoir_control.systems.lake_como.simulation.hydraulic_ppo_study import prepare,worker,evaluate,aggregate,SEEDS,verify,save
from reservoir_control.systems.lake_como.simulation.hydraulic_ppo_mechanism import prepare as prepare_mechanism,run as run_mechanism


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    prepare(args.root)
    prepare_mechanism(args.root)
    save(args.root/'driver.json',dict(source=str(Path(__file__).resolve()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol='Freeze core and supplemental plans before training; two workers, final-only validation, then fixed-state mechanism audit.'))
    plan=json.loads((args.root/'plan.json').read_text());verify(plan)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for record in pool.map(worker,[(args.root/f'seed{s}',s,plan['steps_per_seed']) for s in SEEDS]):
            print(f"Seed {record['seed']}: {record['steps']} steps, failure={record['failure']}",flush=True)
    rows=[]
    for seed in SEEDS:rows.extend(evaluate(args.root,seed,plan))
    aggregate(args.root,rows);run_mechanism(args.root);verify(plan)
    print((args.root/'summary.json').read_text())

if __name__=='__main__':main()
