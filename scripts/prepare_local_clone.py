"""Validate published source/data and prepare a clone-specific audit manifest.

Original run provenance is preserved. Generated egg-info is never scientific
source; Git text may use CRLF or LF without changing its content.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def sha(data): return hashlib.sha256(data).hexdigest()
def git_bytes(repo,ref,path):
    return subprocess.check_output(['git','-c',f'safe.directory={repo.as_posix()}',
        '-C',str(repo),'show',f'{ref}:{path}'])
def normalized(data): return data.replace(b'\r\n',b'\n')

def prepare(root=ROOT):
    destination=root/'LOCAL_SOURCE_MANIFEST.json'
    if destination.exists():raise FileExistsError('Local manifest exists; preserve it before preparing a new one')
    original=root/'SOURCE_MANIFEST.json'
    source=json.loads(original.read_text(encoding='utf-8'))
    files={}; omitted=[]
    for rel,expected in source['copied_files'].items():
        if any(part.endswith('.egg-info') for part in Path(rel).parts):
            omitted.append(rel);continue
        path=root/rel
        if not path.is_file():raise FileNotFoundError(f'Required source or licensed input missing: {rel}')
        current=path.read_bytes()
        if rel.startswith('data/'):
            if sha(current)!=expected:raise ValueError(f'Exact input hash mismatch: {rel}')
        else:
            if rel.startswith('victimagent/'):
                committed=git_bytes(root/'victimagent',source['victimagent_actual_commit'],rel[len('victimagent/'):])
            else:
                committed=git_bytes(root,source['source_commit'],rel)
            if normalized(current)!=normalized(committed):
                raise ValueError(f'Source differs from recorded Git commit: {rel}')
        files[rel]=sha(current)
    actual=subprocess.check_output(['git','-c',f'safe.directory={(root/"victimagent").as_posix()}',
        '-C',str(root/'victimagent'),'rev-parse','HEAD'],text=True).strip()
    if actual!=source['victimagent_actual_commit']:raise ValueError('Wrong victimagent submodule commit')
    payload={**source,'copied_files':files,'original_manifest_sha256':sha(original.read_bytes()),
        'omitted_generated_package_metadata':omitted,
        'copy_policy':'Clone-local byte hashes after comparison with recorded Git source; input hashes remain exact. Original manifest retained.'}
    with destination.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
    return payload

if __name__=='__main__':
    p=prepare();print(json.dumps(dict(status='PASS',validated_files=len(p['copied_files']),
        omitted_generated_metadata=len(p['omitted_generated_package_metadata']),
        manifest=str(ROOT/'LOCAL_SOURCE_MANIFEST.json'))))
