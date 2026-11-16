# SemAudit irregular calibration

This directory contains an executed, locally frozen semi-synthetic calibration. The numerical and text-event timestamp templates come from all eight locally available EPA-Air entities in Time-IMM. **Original numerical values and original narratives are not used.** Numerical trajectories, textual directional propositions, their signs, arrival delays and targets are independently generated.

## Estimands and units

The primary endpoint is the paired increase in test MSE when the same aligned trained predictor receives permuted historical text contents. The permutation rotates the available notes within each seven-day historical query, keeping numerical features, event/arrival times, note counts, cutoffs and targets fixed. History windows do not overlap (seven-day history and stride). With zero or one available note, a content permutation is impossible and the unchanged query is retained. A rotation removes index fixed points but duplicate direction/content can remain; both are explicitly counted.

The predictor is an independent implementation: numerical sufficient statistics plus event-process features and exponentially recency-pooled text features, with a regularized linear head trained by averaged stochastic gradient descent. It is not the original Time-IMM fusion model. The oracle variant receives the generated ±1 direction truth. The sentence variant receives MiniLM embeddings of 16 paired sentences; its PCA center and seven components are fitted on only eight training sentences. Eight different sentences are reserved for validation and test. The sentence experiment tests this restricted template vocabulary; the oracle experiment alone cannot validate language semantics.

Twenty separately generated worlds each contain 128 training, 32 validation and 128 test entities, conditional on the same eight empirical time templates. Each entity has a newly generated numerical trajectory, independent sign process, independent text-template assignment and independent noise. The three SGD training seeds (53, 67, 79) represent optimization variability. Within each entity their losses and then its multiple query losses are averaged before inference. They are not counted as three independent data replications. A one-sided paired t test is applied to 128 generated entity loss differences in each world; rate estimates and Wilson intervals use the 20 independently generated worlds. Inference does not generalize to a random sample of real-world domains.

Signal strengths are 0, 0.25, 0.5 and 1. The target at a declared next-day query is a constructed scalar readout, not a physical continuation of the generated baseline process:

`target = 0.8 * last_numeric + 0.2 * historical_mean + strength * semantic_signal + N(0, 0.35²)`.

For available notes, the semantic signal is the exponential age-weighted sum of independent random ±1 directions divided by the L2 norm of those weights (decay one day). Its conditional variance is one for nonempty histories, and zero for empty histories. Numerical values and the event process are generated independently of the signs. Source entity identities and signs are not given to the timing/numerical comparator. Arrival times are occurrence times plus independently simulated exponential delays with mean 0.15 day. Both occurrence and arrival must precede the cutoff; this latency is simulated, not a claim about real publication availability.

Fixed zero-content evaluation uses the aligned checkpoint. Conventional ablations separately refit a shuffled-input model and a numerical-plus-timing comparator. The validation capability gate is fixed in the configuration: the lower two-sided 95% entity-level CI for improvement over the timing comparator must exceed zero and relative improvement must exceed 1%. All ungated effects and failures remain in the outputs. No hyperparameter or sentence-template selection is performed after reading test scores.

## Freeze and reproduce

Nine structural tests ran before the evaluation. `results/FROZEN_PROTOCOL.json` records the UTC timestamp and SHA256 values of the configuration, implementation, tests and original source files. This is a local protocol freeze; it is **not an externally registered preregistration**. Original EPA-Air and manuscript results were already examined before this experiment; the newly generated outcomes were evaluated after the freeze.

In PowerShell, using the existing project Python environment:

```powershell
& 'F:/CCFA/phase3_prototype/.venv-cuda/Scripts/python.exe' 'F:/CCFA/paper_rewriting_output/journal_extension_20261012/calibration/test_calibration.py'
& 'F:/CCFA/phase3_prototype/.venv-cuda/Scripts/python.exe' 'F:/CCFA/paper_rewriting_output/journal_extension_20261012/calibration/run_calibration.py' --output 'F:/CCFA/paper_rewriting_output/journal_extension_20261012/calibration/reproduction'
& 'F:/CCFA/phase3_prototype/.venv-cuda/Scripts/python.exe' 'F:/CCFA/paper_rewriting_output/journal_extension_20261012/calibration/verify_results.py'
```

The environment name contains CUDA, but this calibration uses CPU only, two threads. Existing scored runs are preserved, and the runner refuses to overwrite `world_results.jsonl`. Reproduction requires the existing empirical source timestamps and the pinned local MiniLM snapshot path in `config.json`.

## Outputs

- `results/summary.json` / `summary.csv`: all eight strength/representation cells, positive/null rejection rates, capability-gated rates, intervals, MSE and effect magnitudes.
- `results/world_results.jsonl`: every independently generated world, including negative effects and validation failures.
- `results/seed_results.csv`: individual training-seed effects, reported as optimization variability.
- `results/entity_loss_results.csv`: losses for each generated test entity, averaged over the three training seeds.
- `results/input_diagnostics.csv`: exact sparse-history and intervention coverage counts.
- `results/raw/world_*.npz`: numerical predictor sufficient statistics, oracle signals, pooled representation values, baseline/noise target components, identity and control arrays. Full numerical trajectories can be regenerated from the script and source timestamps; the archive records the exact features used by the fitted models.
- `results/raw/text_events_*.csv`: event occurrence, simulated arrival, directional truth and phrase index for every generated text event.
- `results/phrase_embeddings.npz`, `encoder_identity.json`: actual embeddings, train-only PCA and full model/tokenizer file hashes.
- `results/models/*.npz`, `model_training_identity.json`: every trained checkpoint, scaling parameters, training seeds, iteration counts and convergence warnings.
- `results/VERIFICATION.json`: all reported test-entity losses independently reconstructed from saved checkpoints with no fitting; all fixed-model controls use the exact same saved aligned parameters.
- `results/run_identity.json`, `execution.log`: environment, run duration, raw/output file hashes and log.

Experimental-design and nature-writing guidance informed design/report organization. The existing extension plan has the verified software attribution: Kassis et al. (2026), *Scientific Agent Skills*, [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065). It is a software/workflow source, not evidence for model performance.
