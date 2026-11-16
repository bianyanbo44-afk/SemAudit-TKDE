# SemAudit TKDE research materials

Research materials for **Representation-Dependent Semantic Reliance in Irregular Text-Augmented Forecasting** by Yanbo Bian and Zhihai Wang.

This repository provides the training and evaluation implementation, experiment configurations, content interventions, generated calibration inputs, task-model checkpoints, paired predictions, verification records, and figure source data. Versioned binary artifacts are available in the `tkde-materials-v1` release. Every release archive and restored research file has a SHA-256 manifest.

## Restore and verify the evidence

Use Python 3.12. Saved-prediction replay requires NumPy. Training and checkpoint inference use the packages in `requirements.txt` and a CUDA-enabled PyTorch installation. The recorded package versions are in `provenance/environment_versions.json`.

```bash
git clone https://github.com/bianyanbo44-afk/SemAudit-TKDE.git
cd SemAudit-TKDE
python -m pip install numpy
python tools/fetch_materials.py --kind training
python tools/verify_materials.py
python experiments/review_followup_20261105/verify_saved_evidence.py
```

The release contains all 1095 journal follow-up checkpoints and 2175 prediction sets. It also includes the 315 added-seed reference fits, their predictions, original reference checkpoints and run configurations, and original reference prediction records. The follow-up comprises 270 replacement-encoder fits, 675 same-family calibration fits, and 150 implementation-variant fits. Historical and fixed-checkpoint comparisons reuse the associated trained checkpoints.

## Obtain benchmark inputs

```bash
python tools/fetch_materials.py --kind data
```

The downloader retrieves the exact five-domain CSV inputs from the official Time-IMM repository at commit `e934cef23d526817dcf0d1a66c6f82274c96c04b` and restores their recorded line endings before verifying their hashes. The release supplies the frozen GPT-2, MiniLM and FinBERT encoding caches used by the experiments. Original data access follows the providers' terms. The pinned source and file records are in `provenance/BENCHMARK_DOWNLOAD_MANIFEST.json`.

## Training and inference

Install `requirements.txt` in the CUDA environment, restore the release, and fetch the benchmark files. Run from the repository root.

```bash
python runtime/run_followup.py test_followup_design.py
python runtime/run_followup.py encoder_coverage.py --phase train
python runtime/run_followup.py strict_history.py
python runtime/run_followup.py neural_calibration.py
python runtime/run_followup.py source_execution.py
python runtime/run_followup.py encoder_history.py
python runtime/run_followup.py verify_checkpoints.py --phase all
python experiments/review_followup_20261105/analyze_followup.py --phase all
```

The original execution sources and manifests are preserved. `runtime/run_followup.py` binds them to the repository-local dependencies. Existing completed fits are reused by the training entry points. To generate fresh fits, move the relevant phase directory to a separate location before running that phase. Keep `results/neural_calibration/inputs` and `phrase_embeddings.pt` when reusing the released generated inputs. The original configurations specify seeds, optimization budgets, preprocessing and stopping rules.

The seven-seed extension of the original three-seed reference is executed by `python runtime/train_reference.py`. This wrapper rebinds three historical path literals in memory and provides the file-digest helper used by the original script. The execution source file retains its recorded hash.

To regenerate MiniLM and FinBERT caches from the raw text, fetch the exact pretrained encoder files and run the encoding phase.

```bash
python tools/fetch_materials.py --kind encoders
python runtime/run_followup.py encoder_coverage.py --phase encode
```

Encoder provider URLs, immutable revisions and file hashes are in `provenance/ENCODER_DOWNLOAD_MANIFEST.json`. The journal generator uses the released MiniLM phrase embeddings and five generated worlds. Its empirical clock-loading dependency is retained in `experiments/added_experiments_20261104/experiments/irregular_fine`. The journal calibration is implemented in `experiments/review_followup_20261105/neural_calibration.py`.

## Figures and manuscript records

`manuscript/source_data` contains the complete figure records and paired summaries. `figure_source` contains the plotting sources and editable vector figures. The article and supplementary manuscript are supplied with the journal submission.

## Provenance and third-party components

`provenance/EXECUTION_SOURCE_MANIFEST.json` identifies the preserved execution sources. `provenance/TRAINING_FILE_MANIFEST.json` identifies the released training files. Original run records retain their execution-time path strings. Repository launchers resolve current dependencies through relative paths.

The upstream IMM-TSF implementation is pinned at `b9e245f2238a9c35d241dbc5bd943c28193a5b90`. Its license and individual file notices are retained under `runtime/phase3_prototype/external/IMM-TSF`. Time-IMM and pretrained encoder inputs are fetched from their providers. Copyright in author-created materials remains with the authors.

Generate all eight figures with `python tools/reproduce_figures.py`. Rebuild their paired records from the restored predictions and downloaded benchmark inputs with `python tools/reproduce_figures.py --rebuild-data`. The launcher binds the retained plotting sources to the current repository layout.

## Authors and funding

Yanbo Bian developed the methodology and software, conducted the experiments, validation and formal analysis, prepared the visualizations, and wrote the original draft. Zhihai Wang contributed to conceptualization and methodology, supervised the research, and reviewed and edited the manuscript. This research received no external funding.

Contact Yanbo Bian at 24722081@bjtu.edu.cn and Zhihai Wang at zhhwang@bjtu.edu.cn.
