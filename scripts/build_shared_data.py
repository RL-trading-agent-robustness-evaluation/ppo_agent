"""Run the pinned victimagent data builder without modifying the submodule."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import tempfile


RAW_NAMES = (
    "crsp_dsf_spy_tlt_gld_2009_2025.csv",
    "stkdistributions_spy_tlt_gld_2009_2025.csv",
    "DGS3MO_2009_2025.csv",
    "DTB3_2009_2025.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    shared = root / "victimagent"
    source = shared / "scripts/build_pipeline.py"
    spec = importlib.util.spec_from_file_location("victimagent_build_pipeline", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned victimagent data pipeline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory(prefix="victim_ppo_data_") as temporary:
        staging = Path(temporary)
        (staging / "configs").mkdir()
        (staging / "data/raw").mkdir(parents=True)
        shutil.copy2(shared / "configs/data_config.yaml", staging / "configs/data_config.yaml")
        for name in RAW_NAMES:
            source_file = root / "data/raw" / name
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            shutil.copy2(source_file, staging / "data/raw" / name)
        status = int(module.build(staging))
        if status != 0:
            raise RuntimeError(f"shared victimagent data pipeline failed with status {status}")
        for relative in ("data/interim", "data/processed", "reports"):
            source_dir = staging / relative
            destination = root / relative
            destination.mkdir(parents=True, exist_ok=True)
            for item in source_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, destination / item.name)
    print("Pinned victimagent data pipeline: PASS")


if __name__ == "__main__":
    main()

