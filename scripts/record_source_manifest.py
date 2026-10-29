"""Record exact unchanged V10 source/data copies; run once when creating V11."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
source=root.parent/'ppo_v10'
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
files={}
for base in ('src','victimagent/src','victimagent/configs','configs'):
    for original in (source/base).rglob('*'):
        if not original.is_file() or '__pycache__' in original.parts or original.suffix=='.pyc':
            continue
        relative=original.relative_to(source)
        target=root/relative
        if target.is_file():
            if digest(original)!=digest(target):
                raise ValueError(f'parent copy changed: {relative}')
            files[relative.as_posix()]=digest(target)
for rel in ('data/processed/v8/spy_market_1993_2025.csv',
            'data/interim/v8/spy_cash_distributions_1993_2025.csv',
            'data/raw/VIXCLS_1990_2026.csv'):
    files[rel]=digest(root/rel)
def commit(path):
    return subprocess.check_output(['git','-c',f'safe.directory={path.as_posix()}',
         '-C',str(path),'rev-parse','HEAD'],text=True).strip()
payload=dict(source_directory='../ppo_v10',source_commit=commit(source),
             victimagent_actual_commit=commit(source/'victimagent'),
             copied_files=files,copy_policy='No Git metadata, environments, checkpoints or experiment results copied.')
with (root/'SOURCE_MANIFEST.json').open('x',encoding='utf-8') as handle:
    json.dump(payload,handle,indent=2)

# Amend only the V11 copies: user explicitly reduced initialization to 200k.
for rel in ('docs/PPO_V11_ADAPTIVE_POLICY_REWARD_SPEC_ZH_HANT.md',
            'docs/prompts/PPO_V11_IMPLEMENTATION_PROMPT_ZH_HANT.md'):
    path=root/rel
    text=path.read_text(encoding='utf-8').replace('v11spec1','v11spec2_200k')
    for old,new in (('500,000','200,000'),('501,760','200,704'),('500000','200000'),('501760','200704'),('500k','200k')):
        text=text.replace(old,new)
    text += '\n\n**2026-10-07 使用者修訂：** 本版本初始化 requested transitions 改為 200,000，SB3 實際 200,704；每次 fine-tuning 仍為 32,768。原 V10 目錄中的 v11spec1 文件保留為歷史版本。\n'
    path.write_text(text,encoding='utf-8')
