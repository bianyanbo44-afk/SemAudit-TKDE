# SemAudit review follow-up evidence

This directory contains the completed exploratory post-review experiments. Natural test results had already been inspected. The locally frozen protocol records identify configurations, code, backend, and corrective execution changes. Every declared cell and capability-screen outcome is retained.

## Completed experiments

| Phase | Trained checkpoints | Saved predictions |
|---|---:|---:|
| MiniLM and FinBERT, all five domains and three families | 270 | 270 |
| Same-family generated irregular signal/null calibration | 675 | 1125 |
| Isolated source-pipeline variants | 150 | 150 |
| Strict historical controls, 150 existing aligned checkpoints | 0 | 450 |
| Historical crossover on all 90 replacement-encoder aligned checkpoints | 0 | 180 |
| Total | 1095 | 2175 |

The calibration includes 450 fixed-model controls in addition to its 675 trained conditions. The historical phase counts aligned reference predictions plus two interventions. Five independent worlds and five signal strengths are crossed with three original families, three paired seeds, and three training conditions. Test entities supply the independent units for entity-level tests. All 75 calibration cells enter the declared Holm families. Encoder support counts are descriptive three-seed paired-win decisions.

## Calibration results

- `immtsf_recavg_gradd` at signal strength 1 has 2/5 Holm-adjusted detections and 2/5 validation-capable worlds.
- `immtsf_t2v_xattn` at signal strength 1 has 3/5 Holm-adjusted detections and 5/5 validation-capable worlds.
- `selective_ssm` at signal strength 1 has 5/5 Holm-adjusted detections and 5/5 validation-capable worlds.

- `immtsf_recavg_gradd` at zero signal has 0/5 nominal and 0/5 adjusted detections.
- `immtsf_t2v_xattn` at zero signal has 0/5 nominal and 0/5 adjusted detections.
- `selective_ssm` at zero signal has 0/5 nominal and 0/5 adjusted detections.

The full response, unscreened and screened outcomes, seed values, and confidence intervals are in `results/analysis`. These measurements use the specified four-channel generator, held-out sentence templates, and eight empirical clock templates.

## Offline prediction replay

From this directory run the following with Python and NumPy.

```text
python verify_saved_evidence.py
```

This checks 2175 prediction sets, finite values, shapes, identifiers, fixed-model target invariants, reconstructed losses, and every file in `EVIDENCE_MANIFEST.json`. CUDA checkpoint reloading was independently executed for all 1095 new trained checkpoints, the 450 generated fixed-model controls, and the 180 replacement-encoder historical interventions. Machine-readable replay records are in `results/verification`.

The parent replay directory retains `verify_artifact.py` for the earlier 468-set evidence and `calibration/verify_results.py` for the 2100 saved linear calibration fits.

## Training and checkpoint-reload environment

The full local archive contains the benchmark checkouts, original run configurations, caches and pretrained encoders required by the original entry points. In that archive, use the recorded CUDA virtual environment and run the following from this directory.

```text
python encoder_coverage.py
python strict_history.py
python source_execution.py
python neural_calibration.py
python analyze_followup.py --phase all
python verify_checkpoints.py --phase all
python encoder_history.py
python verify_checkpoints.py --phase encoder_history
python test_followup_design.py
```

The existing archive layout binds `03_原始代码_数据_训练结果/phase3_prototype` and `05_扩展实验与期刊草稿/journal_extension_20261012`. `SEMAUDIT_PROJECT_ROOT` and `SEMAUDIT_EXTENSION_ROOT` override these roots. The generated calibration also imports the earlier generator under `02_新增实验与数据/SemAudit_added_experiments_20261104`; retain that archive path for full regeneration. Exact source files and model-file identities are recorded in each phase.

The lightweight replay distribution supports offline saved-evidence verification immediately. Full training requires the original project layout, raw benchmark access and corresponding encoder files. Raw third-party CSVs, original text caches and pretrained language-model weights are obtained separately. The new task-model checkpoints and generated inputs are included. The saved inference records specify the CUDA tolerance. CPU/GPU bitwise agreement is outside that verification.

The first source regression attempt had a missing optional import, and concurrent training exhausted GPU memory during the source grid. Corrected execution and resource-resume records retain those events. Completed fits were reused with the same data, seeds and budgets. The empty-history attention guard is covered by explicit finite-gradient and unchanged dense-history tests.
