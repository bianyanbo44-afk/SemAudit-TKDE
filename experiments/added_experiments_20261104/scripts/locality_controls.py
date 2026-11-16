"""Fixed-checkpoint neighborhood intervention and original seed prediction export."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seed_extension", HERE / "scripts/extend_seeds.py")
seedmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seedmod)
audit = seedmod.audit
from eventmamba.timeimm import DATASET_SPECS, GPT2_TEXT_CACHE, GPT2_TEXT_CACHE_1024, load_timeimm
from eventmamba.timeimm_training import prepare_timeimm_split


def local_transform(data, block_size=16, seed=16237):
    windows = data.test_windows
    groups = defaultdict(list)
    for wi, w in enumerate(windows):
        for ei, e in enumerate(w.events):
            if e.modality == data.text_modality:
                groups[data.window_info[w.sample_id].entity].append((e.event_time + data.window_info[w.sample_id].start_time, wi, ei))
    replacements, maps, changed, late = {}, [], 0, 0
    rng = random.Random(seed)
    for entity, events in sorted(groups.items()):
        events.sort()
        for start in range(0, len(events), block_size):
            chunk = events[start:start + block_size]
            shift = rng.randrange(1, len(chunk)) if len(chunk) > 1 else 0
            for i, (recipient_time, wi, ei) in enumerate(chunk):
                donor_time, wj, ej = chunk[(i + shift) % len(chunk)]
                value = windows[wj].events[ej].value
                replacements[wi, ei] = value
                changed += not torch.equal(value, windows[wi].events[ei].value)
                cutoff = data.window_info[windows[wi].sample_id].start_time + windows[wi].cutoff
                late += donor_time > cutoff
                maps.append(dict(entity=entity, recipient_window=windows[wi].sample_id, recipient_event=ei,
                                 donor_window=windows[wj].sample_id, donor_event=ej,
                                 recipient_time=recipient_time, donor_time=donor_time, recipient_cutoff=cutoff))
    altered = audit.replace_values(windows, replacements)
    audit.verify_invariants(windows, altered, data.text_modality)
    return altered, maps, dict(slots=len(maps), changed_vectors=changed, donor_after_cutoff=late,
                               block_size=block_size, seed=seed,
                               offline_intervention_not_deployable=True, all_event_target_window_invariants=True)


def main():
    out = HERE / "results/locality"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    torch.set_num_threads(2)
    device = torch.device("cpu")
    for dataset in DATASET_SPECS:
        for family in seedmod.FAMILIES:
            folder, name = audit.sources_for(dataset, family)
            original = json.loads((folder / "results.json").read_text(encoding="utf8"))
            preprocessing = "layer_norm" if family == "selective_ssm" else "none"
            cache = GPT2_TEXT_CACHE_1024 if dataset == "RepoHealth" else GPT2_TEXT_CACHE
            data = load_timeimm(seedmod.PROJECT / "external/Time-IMM/data" / dataset, DATASET_SPECS[dataset],
                                normalization="train_only", text_cache_name=cache, text_preprocessing=preprocessing)
            altered, maps, diagnostics = local_transform(data)
            audit.write_json(out / f"{audit.SLUGS[dataset]}_{preprocessing}_map.json", maps)
            model = audit.build_model(family, data, original["config"], device)
            packed = {c: prepare_timeimm_split(model, w, data.input_dims, len(data.feature_names))
                      for c, w in {"aligned": data.test_windows, "local_shuffle": altered}.items()}
            for seed in seedmod.OLD:
                checkpoint = folder / f"{name}_seed{seed}.pt"
                original_hash = audit.sha(checkpoint)
                state = torch.load(checkpoint, map_location=device, weights_only=True)
                model.load_state_dict(state["state_dict"], strict=True)
                model_hash = audit.state_digest(model)
                for condition, split in packed.items():
                    metrics, arrays = audit.evaluate(model, split, data, 32)
                    pred = seedmod.predictions(model, split, data)
                    file = out / f"{audit.SLUGS[dataset]}_{family}_{condition}_seed{seed}.npz"
                    np.savez_compressed(file, **arrays, **{k: v for k, v in pred.items() if k not in arrays})
                    if condition == "aligned":
                        assert np.isclose(metrics["mse"], state["test_metrics"]["mse"], rtol=1e-5, atol=1e-6)
                    assert model_hash == audit.state_digest(model)
                    assert original_hash == audit.sha(checkpoint)
                    records.append(dict(dataset=dataset, family=family, seed=seed, condition=condition,
                                        metrics=metrics, diagnostics=diagnostics, checkpoint_sha256=original_hash,
                                        source_checkpoint=str(checkpoint), predictions=str(file)))
                print(f"{dataset} {family} seed={seed} verified", flush=True)
                audit.write_json(out / "results.json", records)
    audit.write_json(out / "VERIFIED.json", dict(evaluations=len(records), expected=90,
                      note="CPU inference; same aligned checkpoints; no fitting; invariant and replay checks passed"))


if __name__ == "__main__":
    main()
