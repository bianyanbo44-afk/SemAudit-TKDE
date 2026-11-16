"""Export pinned source excerpts and paired protocol diagnostics."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
PROTOTYPE = HERE.parents[1] / "phase3_prototype"
EXTENSION = HERE.parent / "journal_extension_20261012"
sys.path.insert(0, str(PROTOTYPE))
sys.path.insert(0, str(PROTOTYPE / "scripts"))
from eventmamba.timeimm import load_timeimm, DATASET_SPECS, GPT2_TEXT_CACHE, GPT2_TEXT_CACHE_1024
from cache_timeimm_text_embeddings import read_entity


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = HERE / "results/source_audit"
    out.mkdir(parents=True, exist_ok=True)
    upstream = PROTOTYPE / "external/IMM-TSF"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    sections = [
        ("Normalization before split", "lib/parse_datasets.py", [(95, 119)]),
        ("Text-dependent membership", "lib/parse_datasets.py", [(203, 230)]),
        ("Cache and raw timestamp origins", "compute_text_embeddings.py", [(64, 76)]),
        ("Raw fallback timestamp origin", "lib/parse_datasets.py", [(153, 171)]),
        ("Query scaling", "lib/parse_datasets.py", [(751, 795)]),
        ("RecAvg subtraction", "fusions/TTF_RecAvg.py", [(85, 100)]),
    ]
    text = ["# Pinned implementation evidence", "", f"IMM-TSF commit `{commit}`.",
            "These excerpts describe this checkout, not every published model or author intent.", ""]
    manifest = []
    for title, relative, ranges in sections:
        path = upstream / relative
        lines = path.read_text(encoding="utf8").splitlines()
        text.extend([f"## {title}", "", f"`{relative}`; SHA256 `{sha(path)}`.", "", "```python"])
        for start, end in ranges:
            text.extend(f"{i}: {lines[i-1]}" for i in range(start, min(end, len(lines))+1))
        text.extend(["```", ""])
        manifest.append(dict(concern=title, file=relative, lines=ranges, sha256=sha(path), commit=commit))
    (out / "SOURCE_EXCERPTS.md").write_text("\n".join(text), encoding="utf8")
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    population = []
    for dataset, spec in DATASET_SPECS.items():
        data = load_timeimm(PROTOTYPE / "external/Time-IMM/data" / dataset, spec,
                            normalization="train_only")
        processed = PROTOTYPE / "external/Time-IMM/data" / dataset / "processed"
        offsets = [read_entity(p, 3600. if spec.time_unit == "hours" else 86400.)["times"][0]
                   for p in sorted(processed.iterdir()) if p.is_dir()]
        population.append(dict(dataset=dataset, window_counts=data.window_counts,
                               dropped_no_text=data.dropped_no_text, timestamp_origin_offsets=offsets,
                               max_abs_offset=float(max(abs(x) for x in offsets)), time_unit=spec.time_unit))
    (out / "population_origin.json").write_text(json.dumps(population, indent=2), encoding="utf8")
    module_path = EXTENSION / "scripts/audit_normalization.py"
    spec = importlib.util.spec_from_file_location("norm_audit", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(module_path), "--prototype-root", str(PROTOTYPE), "--output", str(out / "normalization")]
    module.main()
    print("Pinned excerpts, population/origin checks, physical-prediction scaling sensitivity exported.", flush=True)


if __name__ == "__main__":
    main()
