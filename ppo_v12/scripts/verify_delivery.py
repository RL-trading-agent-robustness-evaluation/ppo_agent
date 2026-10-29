"""Independent stdlib-only verification of delivered files; no PPO imports."""
import hashlib
import json
import math
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'ppo_v11'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
report=ROOT/'reports/v12_benchmark/comparison_2022_2025.json'
result=json.loads(report.read_text(encoding='utf-8'))
assert result['technical_pass'] and result['new_training_transitions']==0 and result['new_model_runs']==0
assert all(sha(Path(p))==h for p,h in result['source_hashes'].items())
assert all(sha(report.parent/name)==h for name,h in result['output_hashes'].items())
orders=json.loads((report.parent/'monthly_purchases.json').read_text(encoding='utf-8'))
assert len(orders)==48 and abs(math.fsum(p['budget'] for p in orders)-1e6)<1e-8
assert {p['month'] for p in orders}=={f'{y}-{m:02}' for y in range(2022,2026) for m in range(1,13)}
rows=[json.loads(s) for s in (report.parent/'monthly_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(rows)==1002 and rows[0]['start_nav']==1e6
assert rows[0]['decision_date']=='2022-01-03' and rows[-1]['valuation_date']=='2025-12-31'
assert math.isclose(rows[-1]['end_nav']/1e6,result['benchmarks']['monthly']['final_wealth'])
assert all(math.isfinite(r['end_nav']) and r['cash_available']>=0 for r in rows)
assert len(result['models'])==5 and [m['seed'] for m in result['models']]==list(range(5))
for name in ('monthly','lump_sum'):
    assert result['wins'][name]==sum(m['metrics']['final_wealth']>result['benchmarks'][name]['final_wealth'] for m in result['models'])
for p in ROOT.joinpath('docs').rglob('*.md'):
    if 'reference' not in p.parts:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if not link.startswith(('https:','http:','#')):
                assert (p.parent/link.split('#')[0]).exists(),(p,link)
spec=ROOT/'docs/PPO_V12_MONTHLY_BH_SPEC_ZH_HANT.md'
assert len(re.findall(r'^## \d+\.',spec.read_text(encoding='utf-8'),re.M))==11
assert sha(ROOT/'docs/reference/PPO_V11_ADAPTIVE_POLICY_REWARD_SPEC_ZH_HANT.md')==sha(PARENT/'docs/PPO_V11_ADAPTIVE_POLICY_REWARD_SPEC_ZH_HANT.md')
assert sha(ROOT/'configs/ppo_v11_reference.yaml')==sha(PARENT/'configs/ppo_v11.yaml')
assert (ROOT/'dev/devlog.md').read_bytes().startswith((PARENT/'dev/devlog.md').read_bytes())
log=(ROOT/'reports/validation/unit_tests_attempt2.log').read_text(encoding='utf-8-sig')
assert 'Ran 9 tests' in log and '\nOK' in log
write(ROOT/'reports/validation/delivery_checks.json',dict(status='PASS',unit_tests=9,
    all_source_and_output_hashes_match=True,all_five_existing_seeds=True,
    source_snapshots_exact=True,devlog_history_preserved=True,spec_sections=11,
    monthly_purchase_count=48,calendar_steps=1002,new_model_runs=0,new_training_transitions=0))
files={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*')
    if p.is_file() and '__pycache__' not in p.parts and p.name!='SOURCE_MANIFEST.json'}
write(ROOT/'SOURCE_MANIFEST.json',dict(version='v12spec1_benchmark_only',files=files,
    parent_root=str(PARENT),referenced_sources=result['source_hashes']))
print(json.dumps(dict(status='PASS',files=len(files),unit_tests=9,wins=result['wins'])))
