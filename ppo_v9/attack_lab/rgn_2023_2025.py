from dataclasses import replace
import argparse
import json
import pandas as pd
import evaluate_v9 as b

parser = argparse.ArgumentParser(description='Evaluate frozen fold-3 PPO V9 models on 2023–2025 with Gaussian noise.')
parser.add_argument('--output', type=b.Path, required=True, help='New output directory for this evaluation')
args = parser.parse_args()
out = args.output.resolve()
b.OUT = out
root = b.ROOT / 'victimagent'
b.runner.v8._set_threads(1)
market, events = b.runner.v8._v8_inputs(root, root, before=None)
fold = next(f for f in b.runner.load_rolling_folds(root/'configs/v8_folds.yaml').folds if f.fold_id == 'fold_3')
window = type(fold.validation)(pd.Timestamp('2023-01-01'), pd.Timestamp('2025-12-31'))
clean, spy = b.runner.v9.v9_bundle(market, events, fit=fold.train, select=window)
source = clean.market_obs
assert source.date.min().year == 2023 and source.date.max() == pd.Timestamp('2025-12-31')
out.mkdir(exist_ok=False)
cfg = json.loads((b.HERE/'config.example.json').read_text())
cfg.update(sigma=0.2,noise_seed=123,columns=list(b.V8_MARKET_FEATURES))
rows, stats = b.prepare.transform(source.to_dict('records'),cfg,'attack')
obs = source.copy()
for col in cfg['columns']:
    obs[col] = [float(r[col]) for r in rows]
assert source.drop(columns=cfg['columns']).equals(obs.drop(columns=cfg['columns']))
assert (obs[cfg['columns']] != source[cfg['columns']]).all().all()
b.save_csv(out/'clean_inputs.csv',source)
b.save_csv(out/'noisy_inputs.csv',obs)
noisy = replace(clean,market_obs=obs)
bh = b.runner.v9._baselines_with_rule(clean,spy)['buy_and_hold']['final_wealth']
results = []
for seed in range(5):
    record = json.loads((b.ROOT/b.runner.RECORDS/f'ppo_v9_gate_relative_fold_3_seed{seed}_v9spec1.json').read_text())
    checkpoint = b.ROOT/record['checkpoint_path']
    assert b.prepare.digest(checkpoint)==record['checkpoint_sha256'] and record['scaler']==b.scaler_fingerprint(clean.observation_builder)
    model=b.PPO.load(checkpoint,device='cpu')
    pair=[b.evaluate(model,v,spy,record['arm'],seed,out/f'seed{seed}_{tag}.csv') for v,tag in [(clean,'clean'),(noisy,'noisy')]]
    a,n=[pd.read_csv(out/r['trajectory_path']).fillna('') for r in pair]
    assert a.decision_date.equals(n.decision_date) and a.gate_forced_full.equals(n.gate_forced_full)
    row=dict(seed=seed,checkpoint=record['checkpoint_sha256'],days=len(a),gate=int(a.gate_forced_full.sum()),clean=pair[0]['metrics']['final_wealth'],noisy=pair[1]['metrics']['final_wealth'])
    for col in ['agent_action','gated_action','executed_action']:
        row[col]=int((a[col]!=n[col]).sum())
    results.append(row)
report=dict(period=['2023-01-01','2025-12-31'],config=cfg,buy_and_hold=bh,results=results)
(out/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
