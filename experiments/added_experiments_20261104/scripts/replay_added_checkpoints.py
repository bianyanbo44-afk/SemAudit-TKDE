"""Independently reload every new neural checkpoint and compare saved predictions."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("extension", HERE / "scripts/extend_seeds.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
from eventmamba.timeimm import load_timeimm, GPT2_TEXT_CACHE, GPT2_TEXT_CACHE_1024
from eventmamba.timeimm_training import prepare_timeimm_split
from eventmamba.multimodal import shuffle_text_values_within_entities, zero_text_values
from eventmamba.training import set_deterministic_seed


def main():
    torch.set_num_threads(2)
    device = torch.device("cuda")
    while not (HERE / "results/neural_encoder/COMPLETE.json").exists():
        time.sleep(15)
    records = []
    for stage in ["ten_seed", "neural_encoder"]:
        while not (HERE / f"results/{stage}/COMPLETE.json").exists():
            time.sleep(15)
        datasets = list(r.DATASET_SPECS) if stage=="ten_seed" else ["FNSPID", "EPA-Air"]
        families = r.FAMILIES if stage=="ten_seed" else ["selective_ssm"]
        seeds = r.SEEDS if stage=="ten_seed" else r.OLD
        for dataset in datasets:
            for family in families:
                folder, _ = r.audit.sources_for(dataset,family)
                config = json.loads((folder/"results.json").read_text())["config"]
                base = r.PROJECT/"external/Time-IMM/data"/dataset if stage=="ten_seed" else HERE/"results/neural_encoder/local_data"/dataset
                cache = (GPT2_TEXT_CACHE_1024 if dataset=="RepoHealth" else GPT2_TEXT_CACHE) if stage=="ten_seed" else "minilm_frozen.pt"
                data = load_timeimm(base, r.DATASET_SPECS[dataset], normalization="train_only", text_cache_name=cache,
                       text_preprocessing="layer_norm" if family=="selective_ssm" else "none")
                sets = dict(aligned=data.test_windows,
                            shuffle=shuffle_text_values_within_entities(data.test_windows,data.window_info,data.text_modality,seed=9103),
                            time_only=zero_text_values(data.test_windows,data.text_modality))
                model = r.audit.build_model(family,data,config,device)
                run = HERE/"results"/stage/r.audit.SLUGS[dataset]
                if stage=="ten_seed":
                    run = run/family
                for condition,windows in sets.items():
                    packed = prepare_timeimm_split(model,windows,data.input_dims,len(data.feature_names))
                    for seed in seeds:
                        set_deterministic_seed(seed)
                        checkpoint = run/f"{condition}_seed{seed}.pt"
                        digest = r.audit.sha(checkpoint)
                        state = torch.load(checkpoint,map_location=device,weights_only=True)
                        model.load_state_dict(state["state_dict"],strict=True)
                        predicted = r.predictions(model,packed,data)
                        z = np.load(run/f"{condition}_seed{seed}_predictions.npz")
                        assert all(np.array_equal(predicted[key],z[key]) for key in ["target","mask","sample_id"])
                        delta = float(np.max(np.abs(predicted["prediction"]-z["prediction"])))
                        assert np.allclose(predicted["prediction"],z["prediction"],rtol=1e-6,atol=1e-6), (stage,dataset,family,seed,condition,delta)
                        assert digest==r.audit.sha(checkpoint)
                        records.append(dict(stage=stage,dataset=dataset,family=family,condition=condition,seed=seed,
                              checkpoint_sha256=digest,maximum_prediction_error=delta,canonical_gpu_replay=True))
                r.save(HERE/"qa/added_checkpoint_replay.json",records)
                print(stage,dataset,family,"replayed",flush=True)
    assert len(records)==333
    r.save(HERE/"qa/ADDED_CHECKPOINTS_VERIFIED.json",dict(checkpoints=len(records),all_replayed=True,
                prediction_tolerance="Independent canonical CUDA replay rtol1e-6, atol1e-6",maximum_absolute_error=max(x["maximum_prediction_error"] for x in records)))


if __name__=="__main__":
    main()
