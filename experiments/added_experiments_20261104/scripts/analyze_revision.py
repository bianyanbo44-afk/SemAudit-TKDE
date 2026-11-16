"""Derive revision tables from all runs, never from manually entered effects."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("extension", HERE / "scripts/extend_seeds.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original(dataset, family):
    folder, name = r.audit.sources_for(dataset, family)
    aligned = read(folder / "results.json")["results"]
    if family == "selective_ssm":
        other = read(r.original_control_file(dataset, family))["results"]
        names = {"shuffle": "selective_ssm_semantic_shuffle", "time_only": "selective_ssm_time_only_zero_text"}
        return {s: {"aligned": next(v for v in aligned if v["model"] == name and v["seed"] == s),
                    **{c: next(v for v in other if v["model"] == model and v["seed"] == s)
                       for c, model in names.items()}} for s in r.OLD}
    suffix = {"aligned": "aligned_unit_consistent", "shuffle": "semantic_shuffle_unit_consistent",
              "time_only": "time_only_unit_consistent"} if family == "immtsf_recavg_gradd" else {
              "aligned": "aligned", "shuffle": "semantic_shuffle", "time_only": "time_only"}
    prefix = "immtsf_recavg_gradd_" if family == "immtsf_recavg_gradd" else "immtsf_t2v_xattn_"
    return {s: {c: next(v for v in aligned if v["seed"] == s and v["model"] == prefix + suffix[c])
                for c in suffix} for s in r.OLD}


def gain(a, b):
    return 100 * (np.mean(b) - np.mean(a)) / np.mean(b)


def holm(ps):
    order = np.argsort(ps)
    adjusted = np.empty(len(ps))
    maximum = 0.
    for rank, i in enumerate(order):
        maximum = max(maximum, (len(ps) - rank) * ps[i])
        adjusted[i] = min(1., maximum)
    return adjusted


def main():
    fits = read(HERE / "results/ten_seed/COMPLETE.json")
    assert fits["fits"] == 315
    out = HERE / "results/analysis"
    out.mkdir(exist_ok=True)
    rows, seedrows, checks, pvalues, checkpoints = [], [], [], [], []
    rng = np.random.default_rng(20261104)
    for dataset in r.DATASET_SPECS:
        for family in r.FAMILIES:
            runs = original(dataset, family)
            for seed in r.SEEDS:
                folder = HERE / "results/ten_seed" / r.audit.SLUGS[dataset] / family
                runs[seed] = {c: read(folder / f"{c}_seed{seed}.json") for c in ["aligned", "shuffle", "time_only"]}
                hashes = {v["initial_state_sha256"] for v in runs[seed].values()}
                assert len(hashes) == 1, (dataset, family, seed, "initialization mismatch")
                pairs = []
                for condition, result in runs[seed].items():
                    checkpoint = Path(result["checkpoint"])
                    checkpoints.append(dict(dataset=dataset, family=family, condition=condition, seed=seed,
                                            checkpoint=str(checkpoint), sha256=r.audit.sha(checkpoint)))
                    z = np.load(folder / f"{condition}_seed{seed}_predictions.npz")
                    err = (z["prediction"].astype(np.float64) - z["target"].astype(np.float64)) ** 2
                    count = z["mask"].sum((0, 1))
                    obs = count > 0
                    metric = ((err * z["mask"]).sum((0, 1))[obs] / count[obs]).mean()
                    delta = abs(float(metric) - result["test_metrics"]["mse"])
                    assert delta < 2e-5 * max(1., metric), (dataset, family, seed, condition, delta)
                    pairs.append((z["target"], z["mask"], z["sample_id"]))
                    checks.append(dict(dataset=dataset, family=family, seed=seed, condition=condition,
                                       prediction_mse_error=delta, same_initialization=True))
                for pair in pairs[1:]:
                    assert all(np.array_equal(a, b) for a, b in zip(pair, pairs[0]))
            seeds = list(runs)
            a, s, t = [np.array([runs[z][c]["test_metrics"]["mse"] for z in seeds])
                       for c in ["aligned", "shuffle", "time_only"]]
            for i, seed in enumerate(seeds):
                seedrows.append(dict(dataset=dataset, family=family, seed=seed,
                    origin="original" if seed in r.OLD else "added", aligned=a[i], shuffle=s[i], time_only=t[i],
                    sag=100*(s[i]-a[i])/s[i], cbtg=100*(t[i]-a[i])/t[i]))
            ix = rng.integers(0, len(seeds), (10000, len(seeds)))
            cintervals = []
            ps = []
            for ctrl in [s, t]:
                boot = 100 * (ctrl[ix].mean(1) - a[ix].mean(1)) / ctrl[ix].mean(1)
                cintervals.append(np.quantile(boot, [.025, .975]).tolist())
                diff = ctrl - a
                positive = int(np.sum(diff > 1e-10))
                nonzero = int(np.sum(np.abs(diff) > 1e-10))
                ps.append(float(binomtest(positive, nonzero, .5, alternative="greater").pvalue) if nonzero else 1.)
            pvalues.extend(ps)
            rows.append(dict(dataset=dataset, family=family, n=len(seeds), aligned=float(a.mean()),
                 aligned_seed_sd=float(a.std(ddof=1)), shuffle=float(s.mean()), time_only=float(t.mean()),
                 sag=gain(a,s), cbtg=gain(a,t), sag_low=cintervals[0][0], sag_high=cintervals[0][1],
                 cbtg_low=cintervals[1][0], cbtg_high=cintervals[1][1],
                 wins_shuffle=int(np.sum(s>a)), wins_time=int(np.sum(t>a)),
                 sign_p_sag=ps[0], sign_p_cbtg=ps[1],
                 old_sag=gain(a[:3],s[:3]), old_cbtg=gain(a[:3],t[:3])))
    adjusted = holm(pvalues)
    for i, row in enumerate(rows):
        row["holm_p_sag"] = float(adjusted[2*i])
        row["holm_p_cbtg"] = float(adjusted[2*i+1])
    sensitivity = []
    for family in r.FAMILIES:
        for margin in [0, .5, 1, 2]:
            for wins in [7, 10]:
                for exclude in [False, True]:
                    subset = [x for x in rows if x["family"] == family and (not exclude or x["dataset"] != "CESNET")]
                    passed = [x["dataset"] for x in subset if x["sag"] > margin and x["cbtg"] > margin
                              and x["wins_shuffle"] >= wins and x["wins_time"] >= wins]
                    sensitivity.append(dict(family=family, margin_percent=margin, required_wins=wins,
                             n_seeds=10, cesnet_excluded=exclude, denominator=len(subset),
                             passed_count=len(passed), passed_datasets=";".join(passed)))
    locality = read(HERE / "results/locality/results.json")
    localrows = []
    for dataset in r.DATASET_SPECS:
        for family in r.FAMILIES:
            subset = [x for x in locality if x["dataset"]==dataset and x["family"]==family]
            a = [x["metrics"]["mse"] for x in subset if x["condition"]=="aligned"]
            b = [x["metrics"]["mse"] for x in subset if x["condition"]=="local_shuffle"]
            diag = subset[0]["diagnostics"]
            localrows.append(dict(dataset=dataset, family=family, aligned=float(np.mean(a)),
                           local_shuffle=float(np.mean(b)), gain=gain(a,b),
                           changed_fraction=diag["changed_vectors"]/diag["slots"],
                           donor_after_cutoff_fraction=diag["donor_after_cutoff"]/diag["slots"]))
    representation = read(HERE / "results/neural_encoder/results.json")
    for result in representation:
        checkpoint = Path(result["checkpoint"])
        checkpoints.append(dict(dataset=result["dataset"], family="selective_ssm_minilm",
                                condition=result["condition"], seed=result["seed"],
                                checkpoint=str(checkpoint), sha256=r.audit.sha(checkpoint)))
    rep_rows = []
    for dataset in ["FNSPID", "EPA-Air"]:
        vals = [[next(x for x in representation if x["dataset"]==dataset and x["seed"]==seed and x["condition"]==c)
                  ["test_metrics"]["mse"] for seed in r.OLD] for c in ["aligned", "shuffle", "time_only"]]
        a, s, t = map(np.array, vals)
        rep_rows.append(dict(dataset=dataset, encoder="MiniLM", family="selective_ssm", n=3,
                         aligned=float(a.mean()), shuffle=float(s.mean()), time_only=float(t.mean()),
                         sag=gain(a,s), cbtg=gain(a,t), wins_shuffle=int(np.sum(a<s)), wins_time=int(np.sum(a<t))))
    for filename, content in [("ten_seed.csv", rows), ("per_seed.csv", seedrows),
                ("threshold_sensitivity.csv", sensitivity), ("locality.csv", localrows),
                ("neural_encoder.csv", rep_rows), ("prediction_checks.csv", checks)]:
        write_csv(out / filename, content)
    r.save(out / "summary.json", dict(ten_seed=rows, rule_sensitivity=sensitivity, locality=localrows,
             neural_encoder=rep_rows, prediction_checks=len(checks),
             inference_scope="Seed bootstrap intervals describe conditional optimization variability on fixed inspected data; 30 exact sign tests receive Holm correction. Neither tests equivalence or unseen-domain semantic absence."))
    r.save(out / "checkpoint_index.json", checkpoints)
    print(json.dumps({"ten_seed_cells": len(rows), "predictions_verified": len(checks),
                      "threshold_cells":len(sensitivity), "neural_encoder_cells":len(rep_rows)}), flush=True)


if __name__ == "__main__":
    main()
