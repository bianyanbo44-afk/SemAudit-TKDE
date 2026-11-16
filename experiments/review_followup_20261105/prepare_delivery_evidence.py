"""Freeze the completed follow-up evidence and write reviewable delivery notes."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT.parent / '08_CameraReady_20261105'


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf8'))


def main():
    completed = read('results/ALL_EVIDENCE_VERIFIED.json')
    assert completed['completed_fits'] == 1095
    assert read('results/verification/encoder_history_checkpoint_replay.json')['fixed_evaluations_checked'] == 180
    for name, expected in [('encoders', 270), ('calibration', 675), ('source', 150)]:
        replay = read(f'results/verification/{name}_checkpoint_replay.json')
        assert replay['verified_checkpoints'] == expected
    calibration = read('results/analysis/neural_calibration_summary.json')
    strongest = [row for row in calibration if row['strength'] == 1.]
    nulls = [row for row in calibration if row['strength'] == 0.]
    results = '\n'.join(
        '- `' + row['family'] + '` at signal strength 1 has ' + str(row['holm_detections']) +
        '/5 Holm-adjusted detections and ' + str(row['validation_capable']) + '/5 validation-capable worlds.'
        for row in strongest)
    null_results = '\n'.join(
        '- `' + row['family'] + '` at zero signal has ' + str(row['nominal_detections']) +
        '/5 nominal and ' + str(row['holm_detections']) + '/5 adjusted detections.' for row in nulls)
    text = '''# SemAudit review follow-up evidence

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

RESULTS

NULL_RESULTS

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
'''.replace('NULL_RESULTS', null_results).replace('RESULTS', results)
    (ROOT / 'README.md').write_text(text, encoding='utf8')
    audit = '''# 审稿意见完成度复核

前轮复核列出的四项实验补充已执行，并已从保存预测和重新加载的模型独立核验。

| 原缺口 | 本轮完成内容 | 解释范围 |
|---|---|---|
| 不规则正控只用了线性头 | 原三个神经族、五世界、五强度（含零信号），675 次训练及 450 次固定模型干预；逐实体推断、能力筛选与 Holm 校正 | 全部检出与未检出保留。功效曲线对应所声明的生成器、时钟、句式及训练预算。 |
| 编码器未覆盖全部神经族和域 | MiniLM、FinBERT 均完成五域三模型族、三条件三种子，270 次训练 | 分别 5/15、6/15 个单元达到原描述性规则。摘要及结论已据此修订。三种子描述性结果与十种子 GPT-2 推断分别报告。 |
| 自然数据局部置换存在未来供体 | 全部 150 个 GPT-2 对齐检查点完成 450 次评估；全部 90 个新编码器对齐检查点再交叉两项历史干预，共 180 次，未来供体数均为零 | 保留事件、内容集合、样本和目标。相似匹配由冻结向量定义。重训比较与固定权重干预分别报告，命题保持编辑尚未进行。 |
| 源码问题缺少真实执行与学习效果矩阵 | 四项回归直接执行固定上游代码修复前后路径；另完成 150 次隔离变体训练和完整受影响单元矩阵 | 所有比较保持共同测试单位，并将尺度变体预测变换回同一训练尺度。结果对应固定上游路径及所评估模型。 |

十种子主审计、阈值/配对胜数敏感性、CESNET 剔除、统计校正、估计对象、域选择说明、符号、文字和图表意见均已落实。完整逐条对应与论文位置见《修改对照与意见落实.md》。

补实验完成后，论文的核心表述更新为表示、学习器和干预共同决定的语义归因。新增强编码器的正结果与原 GPT-2 结果均进入论证。

同族校准在最强信号下，选择性、RecAvg、注意力模型经 Holm 校正分别检出 5/5、2/5、3/5 个生成世界，零信号下名义与调整后检出均为 0/5。完整弱信号曲线、能力筛选和未检出结果均保留。ClusterTrace 注意力模型的编码器重训支持与负的固定历史干预结果并列，防止混淆两项估计对象。

## 仍需作者提供的出版信息

ACM 专属 DOI、ISBN、许可及版权代码尚未提供，`source/rights.tex` 保持待分配字段。作者邮箱、共同通讯、ORCID 和原稿双侧行号均已写入。新增实验属于事后探索性工作。数据覆盖同一基准谱系、同实体时间划分，生成校准覆盖八组时钟和五个生成世界，结论按这些实验范围表述。
'''
    (DELIVERY / '审稿意见完成度复核.md').write_text(audit, encoding='utf8')
    (DELIVERY / 'replay/FOLLOWUP_REPLAY.md').write_text('''# SemAudit evidence replay

Run `python verify_artifact.py` to check the earlier 468 prediction sets and archived file manifest. Run `python followup/verify_saved_evidence.py` to check all 2175 new prediction sets and follow-up file hashes. Both use NumPy and saved evidence. The 1095 new trained task-model checkpoints and generated inputs are under `followup/results`. Independent CUDA reload records accompany them.

Run `python calibration/verify_results.py` for the 2100 saved linear calibration fits. Phase protocols, original pinned support source and regression evidence are retained. Full neural retraining requires the documented archive layout, benchmark data, pretrained encoders and PyTorch environment. Read `followup/README.md` for dependencies and scientific scope.
''', encoding='utf8')
    original_readme = ROOT.parent / '02_新增实验与数据/SemAudit_added_experiments_20261104/experiment_replay/README.md'
    shutil.copy2(original_readme, DELIVERY / 'replay/README.md')
    support = ROOT / 'support/irregular_fine'
    support.mkdir(parents=True, exist_ok=True)
    original_generator = ROOT.parent / '02_新增实验与数据/SemAudit_added_experiments_20261104/experiments/irregular_fine'
    for name in ['run_calibration.py', 'config.json']:
        shutil.copy2(original_generator / name, support / name)
    files = []
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if (not path.is_file() or 'local_data' in relative.parts or '__pycache__' in relative.parts or
                path.suffix == '.pyc' or path.name == 'EVIDENCE_MANIFEST.json'):
            continue
        files.append(dict(file=relative.as_posix(), size=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (ROOT / 'EVIDENCE_MANIFEST.json').write_text(json.dumps(files, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(dict(manifest_files=len(files), new_fits=1095, new_predictions=2175), indent=2))


if __name__ == '__main__':
    main()
