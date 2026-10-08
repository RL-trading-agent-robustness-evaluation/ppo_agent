import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('local_clone',ROOT/'scripts/prepare_local_clone.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def fixture(tmp_path,monkeypatch):
    inputs={'src/a.py':b'value=1\r\n','data/input.csv':b'date,close\n1,2\n'}
    for path,data in inputs.items():
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    (tmp_path/'victimagent').mkdir()
    original=dict(source_commit='parent',victimagent_actual_commit='vendor',
        copied_files={**{p:module.sha(b) for p,b in inputs.items()},'src/pkg.egg-info/PKG-INFO':'generated'})
    (tmp_path/'SOURCE_MANIFEST.json').write_text(json.dumps(original))
    monkeypatch.setattr(module,'git_bytes',lambda *args:b'value=1\n')
    monkeypatch.setattr(module.subprocess,'check_output',lambda *args,**kwargs:'vendor\n')
    return inputs

def test_line_endings_and_generated_metadata(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    before=(tmp_path/'SOURCE_MANIFEST.json').read_bytes()
    result=module.prepare(tmp_path)
    assert len(result['copied_files'])==2
    assert len(result['omitted_generated_package_metadata'])==1
    assert (tmp_path/'SOURCE_MANIFEST.json').read_bytes()==before

def test_rejects_changed_source(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    (tmp_path/'src/a.py').write_bytes(b'value=2\n')
    with pytest.raises(ValueError,match='Source differs'):module.prepare(tmp_path)

def test_rejects_changed_data(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    (tmp_path/'data/input.csv').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='input hash'):module.prepare(tmp_path)

def test_missing_data(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    (tmp_path/'data/input.csv').unlink()
    with pytest.raises(FileNotFoundError):module.prepare(tmp_path)
