"""Exploratory seven-seed extension of the three frozen SemAudit subjects."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1] / "phase3_prototype"
EXTENSION = HERE.parent / "journal_extension_20261012"
sys.path.insert(0, str(PROJECT))
spec = importlib.util.spec_from_file_location("fixed_audit", EXTENSION / "fixed_model/run_fixed_model.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from eventmamba.batching import index_packed_grouped
from eventmamba.timeimm import DATASET_SPECS, GPT2_TEXT_CACHE, GPT2_TEXT_CACHE_1024, load_timeimm
from eventmamba.timeimm_training import PreparedTimeIMMData, prepare_timeimm_split, train_timeimm_model
from eventmamba.multimodal import shuffle_text_values_within_entities, zero_text_values
from eventmamba.training import set_deterministic_seed, append_jsonl

SEEDS = [83, 97, 101, 113, 127, 139, 151]
OLD = [53, 67, 79]
FAMILIES = ["immtsf_recavg_gradd", "immtsf_t2v_xattn", "selective_ssm"]


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")


@torch.inference_mode()
def predictions(model, packed, data):
    model.eval()
    batches = []
    for start in range(0, len(packed.windows), 32):
        ix = torch.arange(start, min(start + 32, len(packed.windows)), device=packed.query_times.device)
        batches.append(model.forward_packed_queries(index_packed_grouped(packed.packed, ix),
                       packed.query_times.index_select(0, ix)).cpu().numpy())
    p = np.concatenate(batches)
    return dict(prediction=p, target=packed.targets.cpu().numpy(), mask=packed.target_masks.cpu().numpy(),
                sample_id=np.array([w.sample_id for w in packed.windows]),
                entity=np.array([data.window_info[w.sample_id].entity for w in packed.windows]),
                start_time=np.array([data.window_info[w.sample_id].start_time for w in packed.windows]))


def original_control_file(dataset, family):
    slug = audit.SLUGS[dataset]
    if family == "selective_ssm":
        return (PROJECT / "results/fnspid_untouched_gate/semantic_controls/results.json"
                if dataset == "FNSPID" else PROJECT / f"results/{slug}_semantic_controls_fresh3_cuda/results.json")
    return audit.sources_for(dataset, family)[0] / "results.json"


def main():
    out = HERE / "results/ten_seed"
    out.mkdir(parents=True, exist_ok=True)
    identity = {"script_sha256": audit.sha(__file__), "seeds_added": SEEDS,
                "original_seeds": OLD, "classification": "exploratory post-acceptance robustness",
                "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                "protocol_sha256": audit.sha(HERE / "REVISION_PROTOCOL.md")}
    save(out / "FROZEN_RUN.json", identity)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    manifest = []
    for dataset in DATASET_SPECS:
        for family in FAMILIES:
            folder, original_name = audit.sources_for(dataset, family)
            original = json.loads((folder / "results.json").read_text(encoding="utf8"))
            config = original["config"]
            training = dict(original.get("effective_training", config["training"]))
            prep = "layer_norm" if family == "selective_ssm" else "none"
            cache = GPT2_TEXT_CACHE_1024 if dataset == "RepoHealth" else GPT2_TEXT_CACHE
            data = load_timeimm(PROJECT / "external/Time-IMM/data" / dataset, DATASET_SPECS[dataset],
                                normalization="train_only", text_cache_name=cache, text_preprocessing=prep)
            windows = {s: getattr(data, s + "_windows") for s in ("train", "val", "test")}
            controls = {"aligned": windows,
                        "shuffle": {s: shuffle_text_values_within_entities(w, data.window_info, data.text_modality,
                                    seed=9101 + i) for i, (s, w) in enumerate(windows.items())},
                        "time_only": {s: zero_text_values(w, data.text_modality) for s, w in windows.items()}}
            run = out / audit.SLUGS[dataset] / family
            run.mkdir(parents=True, exist_ok=True)
            for condition, sets in controls.items():
                probe = audit.build_model(family, data, config, device)
                prepared = PreparedTimeIMMData(**{s: prepare_timeimm_split(probe, w, data.input_dims,
                           len(data.feature_names)) for s, w in sets.items()})
                del probe
                for seed in SEEDS:
                    result_path = run / f"{condition}_seed{seed}.json"
                    if result_path.exists():
                        manifest.append(json.loads(result_path.read_text()))
                        continue
                    set_deterministic_seed(seed)
                    model = audit.build_model(family, data, config, device)
                    initial_hash = audit.state_digest(model)
                    result = train_timeimm_model(name=condition, model=model, data=data, prepared=prepared,
                        output_dir=run, log_record=lambda r: append_jsonl(run / "epochs.jsonl", r),
                        epochs=int(training["epochs"]), patience=int(training["patience"]),
                        batch_size=int(training["batch_size"]), learning_rate=float(training["learning_rate"]),
                        weight_decay=float(training["weight_decay"]), grad_clip=float(training["grad_clip"]), seed=seed)
                    result.update(dataset=dataset, family=family, condition=condition, initial_state_sha256=initial_hash,
                                  training=training, text_preprocessing=prep, source_config=str(folder / "results.json"))
                    arrays = predictions(model, prepared.test, data)
                    np.savez_compressed(run / f"{condition}_seed{seed}_predictions.npz", **arrays)
                    save(result_path, result)
                    manifest.append(result)
                    save(out / "manifest.json", manifest)
                    print(json.dumps({"dataset": dataset, "family": family, "condition": condition,
                          "seed": seed, "mse": result["test_metrics"]["mse"],
                          "completed": len(manifest), "planned": 315}), flush=True)
                    del model
                del prepared
            del data
    save(out / "COMPLETE.json", {"fits": len(manifest), "expected": 315, "finished_utc": time.time()})


if __name__ == "__main__":
    main()
