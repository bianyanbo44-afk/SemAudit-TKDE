"""Replay all original condition-trained checkpoints and export paired predictions."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("revision_analysis", HERE / "scripts/analyze_revision.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
r = m.r
from eventmamba.timeimm import load_timeimm, GPT2_TEXT_CACHE, GPT2_TEXT_CACHE_1024
from eventmamba.timeimm_training import prepare_timeimm_split
from eventmamba.multimodal import shuffle_text_values_within_entities, zero_text_values


def main():
    out = HERE / "results/original_predictions"
    out.mkdir(exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(2)
    manifest = []
    for dataset in r.DATASET_SPECS:
        for family in r.FAMILIES:
            folder, _ = r.audit.sources_for(dataset, family)
            config = m.read(folder / "results.json")["config"]
            data = load_timeimm(r.PROJECT / "external/Time-IMM/data" / dataset, r.DATASET_SPECS[dataset],
                 normalization="train_only", text_cache_name=GPT2_TEXT_CACHE_1024 if dataset=="RepoHealth" else GPT2_TEXT_CACHE,
                 text_preprocessing="layer_norm" if family=="selective_ssm" else "none")
            variants = dict(aligned=data.test_windows,
                 shuffle=shuffle_text_values_within_entities(data.test_windows, data.window_info, data.text_modality, seed=9103),
                 time_only=zero_text_values(data.test_windows, data.text_modality))
            model = r.audit.build_model(family, data, config, device)
            packed = {c: prepare_timeimm_split(model, w, data.input_dims, len(data.feature_names)) for c,w in variants.items()}
            for seed, conditions in m.original(dataset, family).items():
                for condition, result in conditions.items():
                    path = Path(result["checkpoint"])
                    state = torch.load(path, map_location=device, weights_only=True)
                    model.load_state_dict(state["state_dict"], strict=True)
                    metrics, _ = r.audit.evaluate(model, packed[condition], data, 32)
                    assert np.isclose(metrics["mse"], result["test_metrics"]["mse"], rtol=1e-5, atol=1e-6)
                    name = f"{r.audit.SLUGS[dataset]}_{family}_{condition}_seed{seed}.npz"
                    np.savez_compressed(out / name, **r.predictions(model, packed[condition], data))
                    manifest.append(dict(dataset=dataset, family=family, condition=condition, seed=seed,
                           mse=metrics["mse"], checkpoint_sha256=r.audit.sha(path), predictions=name,
                           source_checkpoint=str(path), replay_verified=True))
            r.save(out / "manifest.json", manifest)
            print(dataset, family, "replayed", flush=True)
    r.save(out / "VERIFIED.json", dict(evaluations=len(manifest), expected=135))


if __name__ == "__main__":
    main()
