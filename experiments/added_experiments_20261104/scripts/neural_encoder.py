"""Full-note MiniLM sensitivity in the original selective architecture."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seed_extension", HERE / "scripts/extend_seeds.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
audit = runner.audit
sys.path.insert(0, str(runner.PROJECT / "scripts"))
from cache_timeimm_text_embeddings import read_entity
from eventmamba.timeimm import load_timeimm, DATASET_SPECS
from eventmamba.multimodal import shuffle_text_values_within_entities, zero_text_values
from eventmamba.timeimm_training import PreparedTimeIMMData, prepare_timeimm_split, train_timeimm_model
from eventmamba.training import set_deterministic_seed, append_jsonl


def main():
    # Do not contend for GPU memory with the seed extension.
    while not (HERE / "results/ten_seed/COMPLETE.json").exists():
        time.sleep(15)
    out = HERE / "results/neural_encoder"
    out.mkdir(parents=True, exist_ok=True)
    source = runner.EXTENSION / "external_validation/models/minilm"
    runner.save(out / "FROZEN_RUN.json", dict(model="sentence-transformers/all-MiniLM-L6-v2",
                revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", seeds=runner.OLD,
                datasets=["FNSPID", "EPA-Air"], max_length=256, all_notes=True,
                script_sha256=audit.sha(__file__), classification="exploratory post-acceptance sensitivity"))
    torch.set_num_threads(4)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(source, local_files_only=True, trust_remote_code=False).to(device).eval()
    records = []
    for dataset in ["FNSPID", "EPA-Air"]:
        raw = runner.PROJECT / "external/Time-IMM/data" / dataset / "processed"
        for entity in sorted(p for p in raw.iterdir() if p.is_dir()):
            target = out / "local_data" / dataset / "processed" / entity.name
            target.mkdir(parents=True, exist_ok=True)
            for filename in ["text.csv", "time_series.csv"]:
                shutil.copy2(entity / filename, target / filename)
            cache = target / "minilm_frozen.pt"
            if cache.exists():
                continue
            parsed = read_entity(entity, 86400.)
            vectors = []
            with torch.inference_mode():
                for start in range(0, len(parsed["notes"]), 64):
                    batch = tokenizer(parsed["notes"][start:start+64], padding=True, truncation=True,
                                      max_length=256, return_tensors="pt").to(device)
                    hidden = encoder(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                    vectors.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
            embeddings = torch.cat(vectors)
            torch.save(dict(embeddings=embeddings,
                       relative_times=torch.tensor(parsed["times"], dtype=torch.float64),
                       source_row_indices=torch.tensor(parsed["row_indices"], dtype=torch.int64),
                       metadata=dict(timestamp_base="first numeric timestamp", embedding_dim=384,
                                     text_sha256=parsed["text_sha256"], pooling="masked mean L2",
                                     max_length=256, model_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")), cache)
            print(f"encoded {dataset}/{entity.name}: {len(embeddings)} notes", flush=True)
    del encoder
    torch.cuda.empty_cache()
    for dataset in ["FNSPID", "EPA-Air"]:
        data = load_timeimm(out / "local_data" / dataset, DATASET_SPECS[dataset], normalization="train_only",
                            text_cache_name="minilm_frozen.pt", text_preprocessing="layer_norm")
        folder, _ = audit.sources_for(dataset, "selective_ssm")
        config = json.loads((folder / "results.json").read_text())["config"]
        original = {s: getattr(data, s + "_windows") for s in ["train", "val", "test"]}
        variants = dict(aligned=original,
             shuffle={s: shuffle_text_values_within_entities(w, data.window_info, data.text_modality, seed=9101+i)
                      for i, (s, w) in enumerate(original.items())},
             time_only={s: zero_text_values(w, data.text_modality) for s, w in original.items()})
        run = out / audit.SLUGS[dataset]
        run.mkdir(exist_ok=True)
        for condition, windows in variants.items():
            model = audit.build_model("selective_ssm", data, config, device)
            packed = PreparedTimeIMMData(**{s: prepare_timeimm_split(model, w, data.input_dims, len(data.feature_names))
                                           for s, w in windows.items()})
            del model
            for seed in runner.OLD:
                file = run / f"{condition}_seed{seed}.json"
                if file.exists():
                    records.append(json.loads(file.read_text()))
                    continue
                set_deterministic_seed(seed)
                model = audit.build_model("selective_ssm", data, config, device)
                training = config["training"]
                result = train_timeimm_model(name=condition, model=model, data=data, prepared=packed,
                     output_dir=run, log_record=lambda r: append_jsonl(run / "epochs.jsonl", r),
                     epochs=training["epochs"], patience=training["patience"], batch_size=training["batch_size"],
                     learning_rate=training["learning_rate"], weight_decay=training["weight_decay"],
                     grad_clip=training["grad_clip"], seed=seed)
                result.update(dataset=dataset, condition=condition, encoder="MiniLM", input_width=384)
                runner.save(file, result)
                np.savez_compressed(run / f"{condition}_seed{seed}_predictions.npz",
                                    **runner.predictions(model, packed.test, data))
                records.append(result)
                runner.save(out / "results.json", records)
                print(json.dumps({"dataset": dataset, "condition": condition, "seed": seed,
                                  "mse": result["test_metrics"]["mse"]}), flush=True)
                del model
    runner.save(out / "COMPLETE.json", dict(fits=len(records), expected=18))


if __name__ == "__main__":
    main()
