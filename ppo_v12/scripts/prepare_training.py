"""Create an isolated V11 engine copy for authorized V12 500k training."""
import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'ppo_v11'
ENGINE=ROOT/'training'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if ENGINE.exists(): raise FileExistsError('training checkout already exists; never reset it')
ENGINE.mkdir()
manifest=json.loads((PARENT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
for rel,expected in manifest['copied_files'].items():
    source=PARENT/rel
    assert sha(source)==expected,rel
    target=ENGINE/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
changes={}
for base in ('src/victim_ppo/v11','tests'):
    for source in (PARENT/base).glob('*.py'):
        target=ENGINE/source.relative_to(PARENT)
        target.parent.mkdir(parents=True,exist_ok=True)
        text=source.read_text(encoding='utf-8')
        text=text.replace('ppo_v11_', 'ppo_v12_').replace("artifacts/ppo_v11", "artifacts/ppo_v12")
        text=text.replace('v11spec2_200k','v12train1_500k')
        text=text.replace('200704','501760').replace('200000','500000')
        target.write_text(text,encoding='utf-8')
        changes[str(target.relative_to(ENGINE))]=dict(source_sha256=sha(source),v12_sha256=sha(target))
config=ENGINE/'configs/ppo_v11.yaml'
config.write_text((PARENT/'configs/ppo_v11.yaml').read_text(encoding='utf-8')
    .replace('v11spec2_200k','v12train1_500k').replace('200000','500000').replace('200704','501760'),encoding='utf-8')
for rel in ('scripts/run_ppo_v11.py','docs/PPO_V11_ADAPTIVE_POLICY_REWARD_SPEC_ZH_HANT.md',
            'docs/prompts/PPO_V11_IMPLEMENTATION_PROMPT_ZH_HANT.md'):
    target=ENGINE/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    text=(PARENT/rel).read_text(encoding='utf-8')
    if rel.endswith('.py'):
        text=text.replace('ppo_v11_','ppo_v12_').replace('200k','500k')
    else:
        text='> V12 training override: see ../../../docs/PPO_V12_FULL_TRAINING_ADDENDUM_ZH_HANT.md. Initial request 500000 / actual 501760; five seeds 0–4. All other V11 economics remain unchanged.\n\n'+text
    target.write_text(text,encoding='utf-8')
manifest['source_directory']=str(PARENT)
manifest['copy_policy']='Exact original copied_files preserved; V11 engine adapted separately in ENGINE_CHANGES.json; no old artifacts/models copied.'
(ENGINE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
changes['config']=dict(source_sha256=sha(PARENT/'configs/ppo_v11.yaml'),v12_sha256=sha(config))
(ENGINE/'ENGINE_CHANGES.json').write_text(json.dumps(dict(changes=changes,
    rationale='Isolated version/namespace and initialization-budget change only; preserve reward, schedule, fresh rollouts, execution and account contracts.'),indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',engine=str(ENGINE),initial_requested=500000,initial_expected=501760)))
