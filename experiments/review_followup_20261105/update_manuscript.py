"""Integrate verified follow-up evidence in the accepted review layout."""
from pathlib import Path
import difflib
import json
import re
import runpy
import shutil

ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT.parent / '08_CameraReady_20261105'
ANALYSIS = ROOT / 'results/analysis'


def read(name):
    return json.loads((ANALYSIS / name).read_text(encoding='utf8'))


def marked(text):
    return '\\revised{' + text + '}'


def percentage(value, digits=3):
    return '$' + format(value, '+.' + str(digits) + 'f') + '$'


def main():
    assert (ROOT / 'results/ALL_EVIDENCE_VERIFIED.json').is_file()
    encoders = read('encoder_summary.json')
    calibration = read('neural_calibration_summary.json')
    history = read('strict_history_summary.json')
    source = read('source_effect_summary.json')
    encoder_history = read('encoder_history_summary.json')
    supports = {name: sum(row['joint_support'] for row in encoders if row['encoder'] == name)
                for name in ['minilm', 'finbert']}
    backup = DELIVERY / 'qa/before_followup_experiments'
    if not backup.exists():
        backup.mkdir(parents=True)
        for name in ['SemAudit_CameraReady_WithLineNumbers.pdf', 'SemAudit_Marked_WithLineNumbers.pdf',
                     'SemAudit_Source.zip', '修改对照与意见落实.md', '审稿意见完成度复核.md']:
            shutil.copy2(DELIVERY / name, backup / name)
        shutil.copy2(DELIVERY / 'source/main.tex', backup / 'main.tex')
    runpy.run_path(str(DELIVERY / 'tools/revise_manuscript.py'), run_name='__main__')
    path = DELIVERY / 'source/main.tex'
    paper = path.read_text(encoding='utf8')
    changes = json.loads((DELIVERY / 'qa/change_blocks.json').read_text(encoding='utf8'))

    def change(old, new, reason):
        nonlocal paper
        assert paper.count(old) == 1, (reason, paper.count(old))
        paper = paper.replace(old, new)
        changes.append(dict(reason=reason, before=old, after=new))

    def paragraph(needle, new, reason):
        candidates = [block for block in paper.split('\n\n') if needle in block]
        assert len(candidates) == 1, needle
        change(candidates[0], marked(new), reason)

    def subsection(label, new, reason):
        start = paper.index('\\label{' + label + '}')
        content_start = paper.index('\n\n', start) + 2
        following = re.search(r'\\(?:subsection|section)\{', paper[content_start:])
        assert following is not None
        end = content_start + following.start()
        change(paper[content_start:end], new.rstrip() + '\n\n', reason)

    abstract = r'''Web and operational platforms increasingly forecast numerical outcomes from irregular event streams that pair measurements with asynchronous text. Yet predictive gains leave unresolved whether models use text meaning, its occurrence process, or preprocessing cues. We introduce \semaudit, a counterfactual protocol that preserves forecast units and event metadata while separating alignment, content beyond timing, and multimodal utility. An independent Fidel-TS control yields a 25.45\% aligned-over-shuffle gain, and a same-family irregular signal/null calibration recovers injected alignment with model-dependent sensitivity. On five Time-IMM datasets, a ten-seed GPT-2 audit finds no joint alignment-and-content support under the declared paired-win rules. Replacing GPT-2 throughout the same three neural families changes the result. Three-seed MiniLM and FinBERT audits support MINILM/15 and FINBERT/15 dataset--family cells, respectively, exposing representation-dependent attribution. A frozen financial-news gate further separates ordinary model-comparison wins from semantic support. Historical-only interventions and executed source-pipeline comparisons measure these effects on common forecast units. Together, the experiments show how semantic attribution depends on the tested representation, learning procedure, and intervention, and provide a reproducible way to assess that dependence.'''
    abstract = abstract.replace('MINILM', str(supports['minilm'])).replace('FINBERT', str(supports['finbert']))
    match = re.search(r'\\begin\{abstract\}\n(.*?)\n\\end\{abstract\}', paper, re.S)
    change(match[1], marked(abstract), 'Expanded evidence changes the abstract to representation-dependent attribution')
    paragraph('We calibrate diagnostic sensitivity with a frozen rule',
        r'''We calibrate sensitivity on an independent Fidel-TS photovoltaic task with scheduled weather text \cite{xu2025fidelts}. Aligned TGTSF improves 25.45\% over shuffled reports with three of three paired wins. A locally frozen FNSPID gate then shows that ordinary model-comparison wins can coexist with failure of semantic controls. We extend the original GPT-2 audit to ten seeds, replace its encoder across every domain and family, and calibrate the same neural families on generated historical irregular streams. The encoder replacements recover positive support in configurations that the shared GPT-2 representation did not support, making representation choice part of the attribution question.''',
        'Introduction incorporates encoder reversals')
    paragraph('The protocol turns an attribution claim into an auditable comparison',
        r'''The protocol makes an attribution claim auditable by requiring aligned content to contribute beyond reassignment and the preserved occurrence process. Sensitivity calibration and full-grid encoder replacements determine the configurations in which that evidence appears.''',
        'Concise constructive contribution')
    change(r'''\item We validate diagnostic sensitivity on an independent positive control, then conduct a frozen three-family stress test over five datasets spanning financial news, software repositories, computing systems, network traffic, and environmental monitoring. A preregistered financial-news gate demonstrates that ordinary multimodal gains can survive while semantic necessity fails.''',
        r'''\item \revised{We calibrate sensitivity with natural and same-family irregular signal/null controls. Across five domains and three neural families, a ten-seed GPT-2 audit and complete MiniLM/FinBERT replacements show representation-dependent attribution. A locally frozen financial-news gate separates ordinary model-comparison wins from semantic support.}''',
        'Contribution includes same-family calibration and the full encoder grid')
    change('The audit identifies four protocol couplings and quantifies one implementation repair at the same scale as the claimed semantic effects.',
        'Executed upstream regressions and 150 isolated-variant fits quantify four protocol couplings on common forecast units.',
        'Contribution reports all executed source-pipeline effects')
    change('MiniLM checks use the original three seeds.',
        'Full-grid MiniLM and FinBERT replacements use the original three paired seeds for all five domains and three families.',
        'Setup reports the complete encoder coverage')

    order = ['selective_ssm', 'immtsf_recavg_gradd', 'immtsf_t2v_xattn']
    strongest = {row['family']: row for row in calibration if row['strength'] == 1.}
    nulls = {row['family']: row for row in calibration if row['strength'] == 0.}
    detections = ', '.join(str(strongest[family]['holm_detections']) for family in order)
    null_detections = ', '.join(str(nulls[family]['nominal_detections']) for family in order)
    capability = ', '.join(str(strongest[family]['validation_capable']) for family in order)
    cal_text = marked(r'''We calibrate the original three neural families on generated historical text streams using the empirical numerical and text clocks of eight EPA-Air entities. Four numerical channels retain that task's output width. Independent directions encoded by actual frozen MiniLM sentence embeddings contribute to targets through $\lambda z(1,-1,.5,-.5)$, added to $0.8v_{\rm last}+0.2\bar v+\epsilon$ with noise SD 0.35. Arrival delays have mean 0.15 day, and all text must arrive by the forecast cutoff. Train and held-out phrase templates are disjoint. Signal strengths are $0,.05,.1,.25,1$.''') + '\n\n'
    cal_text += marked(r'''Five generated worlds each contain 64 training, 16 validation, and 64 test entities, with four non-overlapping seven-day histories per entity. The original model widths, optimization budgets, and three paired seeds yield 675 fits across aligned, within-history rotation, and time-only conditions. The aligned checkpoints are additionally evaluated under rotation and zero content. Seed losses are averaged within each entity before exact directional sign tests and entity-bootstrap intervals. Holm correction covers 75 world--strength--family tests per contrast. A validation capability screen requires $>1\%$ gain over the separately trained time-only model, while all unscreened results remain reported.''') + '\n\n'
    cal_text += marked(r'''At $\lambda=1$, adjusted alignment detection occurs in DETECTIONS of five worlds for the selective, RecAvg, and attention families, respectively. Validation capability holds in CAPABILITY of five worlds. Nominal null detections are NULLS of five. Figure~\ref{fig:irregular} gives the full response and uncertainty, while source data include every world, strength, seed, and capability-screen outcome. This measures sensitivity of the audited neural families under the declared generator. The existing 20-world linear calibration over seven strengths remains available as a complementary estimator check.'''.replace('DETECTIONS', detections).replace('CAPABILITY', capability).replace('NULLS', null_detections)) + '\n\n'
    cal_text += r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_neural_calibration.pdf}
\caption{\revised{Same-family irregular signal/null calibration. (a) Fixed-checkpoint alignment gain across the declared strength grid, with means and 95\% $t$ intervals across five generated worlds. (b) Detection after Holm correction across 75 world--strength--family tests, with Wilson 95\% intervals across worlds. Three optimization seeds are averaged within each independent test entity. All 75 generated cells are included.}}
\label{fig:irregular}
\Description{Two panels show alignment effects and multiplicity-adjusted detection rates for the original three neural model families across five injected signal strengths. Colors, markers and line styles distinguish the families.}
\end{figure*}'''
    subsection('sec:irregular-calibration', cal_text, 'Direct calibration of the original neural families')
    paragraph("A full-note MiniLM replacement changes the selective model's input width",
        r'''Replacing GPT-2 across all five domains and three neural families yields representation-dependent support (Table~\ref{tab:encoders}). Under the original zero-margin, two-of-three-win rule, MiniLM supports MINILM/15 cells and FinBERT supports FINBERT/15. FinBERT on ClusterTrace has positive SAG and CBTG in all three families. Its attention-family gains are 2.373\% and 4.270\%, with three of three wins against both controls. MiniLM on that model gives 3.736\% and 1.546\%, with two of three wins. These are three-seed descriptive decisions. The same attention checkpoints have historical-rotation gains of $-0.191\%$ for MiniLM and $-0.003\%$ for FinBERT, each with one of three wins. Encoder choice changes the condition-retrained outcome, while fixed-model dependence requires its own contrast.'''.replace('MINILM', str(supports['minilm'])).replace('FINBERT', str(supports['finbert'])),
        'Report the complete encoder matrix and positive findings')
    rotation = [row for row in history if row['condition'] == 'history_rotation']
    nearest = [row for row in history if row['condition'] == 'history_nearest_swap']
    historical_text = r'''We evaluate all 150 aligned GPT-2 checkpoints under two within-history interventions. Temporal rotation reassigns content in chronological blocks of at most four notes. Nearest-content swapping pairs the most similar remaining note vectors within the recipient history. Both preserve its content multiset, event times/counts, numerical inputs, masks, targets, and forecast windows. Every donor belongs to the same entity and split and arrives by the recipient cutoff, giving zero future donors across all 450 evaluations. Zero/singleton histories are retained.'''
    historical_text += '\n\n' + r'''Across 15 dataset--family cells, temporal-rotation gains range from ROTLOW\% to ROTHIGH\%, and nearest-content gains from NEARLOW\% to NEARHIGH\%. Actual vector-change rates span 43.75--100\% and 37.5--97.48\%, respectively. The similarity restriction uses the frozen representation, so achieved coverage and intervention scope accompany every contrast. Archived predictions and the new checkpoint replays also record numerical resolution separately from paired optimization variability.'''
    for token, values, function in [('ROTLOW', rotation, min), ('ROTHIGH', rotation, max),
                                     ('NEARLOW', nearest, min), ('NEARHIGH', nearest, max)]:
        historical_text = historical_text.replace(token, percentage(function(row['gain'] for row in values), 6))
    historical_text += '\n\n' + r'''Crossing both interventions with all 90 aligned replacement-encoder checkpoints adds 180 fixed-model evaluations. Their 60 dataset--family--encoder--intervention cells have gains from LOW\% to HIGH\%, with zero future donors. The complete crossover accompanies the condition-retrained comparisons in Table~\ref{tab:encoders}.'''.replace(
        'LOW', percentage(min(row['fixed_gain'] for row in encoder_history), 6)).replace(
        'HIGH', percentage(max(row['fixed_gain'] for row in encoder_history), 6))
    subsection('sec:locality', '\n\n'.join(marked(block) for block in historical_text.split('\n\n')),
               'Historical-only interventions replace the future-donor stress test')
    paragraph('provides an operational distinction between multimodal utility',
        r'''\semaudit{} separates three questions that aggregate multimodal loss conflates. The frozen financial-news gate demonstrates their different decisions under GPT-2. The ten-seed extension retains that representation-specific result, while full-grid encoder replacements recover support in additional configurations, including cross-family directional agreement on ClusterTrace with FinBERT. The same-family generated calibration measures the response to controlled signal strength. Together, these comparisons make the representation, learner, and intervention explicit components of a semantic-attribution claim.''',
        'Discussion incorporates encoder-dependent support')
    paragraph('The frozen GPT-2 mean embedding may omit task-relevant propositions',
        r'''The encoder replacements show that the original cross-family decision depends on the shared representation. Conditional redundancy and optimization can further influence modality use \cite{zhang2025when,wu2022greedy,peng2022ogm,chaudhuri2025collapse}. Reporting the representation alongside event-preserving controls makes the measured attribution conditional on a reproducible forecasting configuration.''',
        'Representation explanation is supported by controlled replacements')
    paragraph('The natural positive control is regular with scheduled future context',
        r'''The natural photovoltaic control uses scheduled weather context, while the generated neural calibration uses four-channel targets, held-out sentence templates, five worlds, and eight empirical clock templates. Its measured sensitivity is conditional on that generator. The natural data cover one benchmark lineage and same-entity chronological splits. Encoder replacements use three seeds, and similarity-matched interventions constrain embedding similarity rather than proposition equivalence. The original attention width remains 128. These conditions define the scope of the reported attribution evidence.''',
        'Update scope after the additional experiments')
    paragraph('makes semantic attribution in irregular text-augmented forecasting an executable',
        r'''\semaudit{} turns semantic attribution into an executable comparison over fixed forecast units and event metadata. The frozen gate separates multimodal gains from alignment-dependent evidence, and the expanded encoder grid shows where that evidence changes with the representation. Same-family signal/null calibration, historical-only interventions, and executed source-pipeline comparisons provide the measurements needed to interpret those outcomes. This protocol gives forecasting studies a reproducible way to test the content property used to explain their gains.''',
        'Conclusion reflects the expanded evidence')

    family_order = ['selective_ssm', 'immtsf_recavg_gradd', 'immtsf_t2v_xattn']
    encoder_lines = []
    for encoder in ['minilm', 'finbert']:
        for dataset in ['ClusterTrace', 'CESNET', 'EPA-Air', 'FNSPID', 'RepoHealth']:
            cells = [next(row for row in encoders if row['encoder'] == encoder and row['dataset'] == dataset and row['family'] == family)
                     for family in family_order]
            values = [percentage(row['sag'], 4) + '/' + percentage(row['cbtg'], 4) +
                      ' (' + str(row['wins_shuffle']) + '/' + str(row['wins_time']) + ')' for row in cells]
            encoder_lines.append(dataset + ' & ' + ('MiniLM' if encoder == 'minilm' else 'FinBERT') + ' & ' + ' & '.join(values) + r' \\')
    encoder_table = r'''\begin{table*}[t]
\caption{\revised{Complete frozen-encoder neural replacement grid, three paired seeds per cell. Each cell reports SAG/CBTG in percent, followed by aligned wins against shuffle/time-only. Internal model widths and training budgets match the original families. Full condition losses, token diagnostics, and predictions accompany the source data.}}
\label{tab:encoders}
\centering
\small
\ifrevision\color{red!75!black}\fi
\begin{tabular}{llccc}
\toprule
Dataset & Encoder & Selective state-space & RecAvg + GR-Add & T2V-XAttn + XAttn-Add \\
\midrule
ENCODER_ROWS
\bottomrule
\end{tabular}
\end{table*}'''.replace('ENCODER_ROWS', '\n'.join(encoder_lines))
    encoder_details = marked(r'''The frozen encoders are \texttt{sentence-transformers/all-MiniLM-L6-v2} and \texttt{ProsusAI/finbert} \cite{minilm_modelcard,finbert_modelcard}. Their replacements across five domains and three families yield 270 fits over three matched conditions and seeds 53, 67, and 79. Masked-mean pooling and L2 normalization use token limits of 256 and 512, respectively. All note slots are retained, with per-note truncation and exact local model-file hashes recorded. The existing input projection accepts each encoder width while internal RecAvg/attention widths, budgets, and validation selection remain fixed. Independent residual-ridge checks remain separately archived under their own selection and aggregation.''')
    subsection('app:encoder', encoder_table + '\n\n' + encoder_details,
               'Every modern-encoder cell is included in the appendix')
    variants = ['full_entity_normalization', 'cache_text_origin', 'retain_empty_training_windows']
    source_lines = []
    for dataset in ['ClusterTrace', 'CESNET', 'EPA-Air', 'FNSPID', 'RepoHealth']:
        for family in family_order:
            cells = [next(row for row in source if row['dataset'] == dataset and row['family'] == family and row['variant'] == variant)
                     for variant in variants]
            name = {'selective_ssm': 'Selective', 'immtsf_recavg_gradd': 'RecAvg', 'immtsf_t2v_xattn': 'T2V-XAttn'}[family]
            source_lines.append(dataset + ' & ' + name + ' & ' + ' & '.join(percentage(row['canonical_scale_change_percent']) for row in cells) + r' \\')
    source_table = r'''\begin{table*}[t]
\caption{\revised{Isolated source-pipeline effects on common canonical test windows, three paired seeds. Entries are percentage changes in MSE relative to the canonical path. Predictions are transformed back to the same train-only numerical scale before comparison. Full-entity normalization, cached text origin, and empty-history training inclusion vary individually. Positive entries mean higher loss under the variant.}}
\label{tab:source-effects}
\centering
\small
\ifrevision\color{red!75!black}\fi
\begin{tabular}{llrrr}
\toprule
Dataset & Family & Full-entity scale & Text-based origin & Include empty train histories \\
\midrule
SOURCE_ROWS
\bottomrule
\end{tabular}
\end{table*}'''.replace('SOURCE_ROWS', '\n'.join(source_lines))
    paragraph('The pinned Time-IMM/IMM-TSF paths exhibit four observable couplings',
        r'''The pinned source paths couple normalization, no-text window membership, cached/raw time origins, and RecAvg time units. Executed upstream regressions verify each path before and after its targeted repair. A 150-fit grid then isolates learned effects on common forecast units (Table~\ref{tab:source-effects}). In train-only evaluation coordinates, full-entity-scaling variants change loss by $-56.56$ to $+14.94\%$, cached-origin variants by $-3.82$ to $+2.97\%$, and empty-history training inclusion by $-6.80$ to $+2.23\%$. The complete matrix identifies the affected evaluated model/domain configurations.''',
        'All four source issues have executed and quantified evidence')
    anchor = '\\label{app:repro}'
    change(anchor, anchor + '\n\n' + source_table, 'Include the complete learned source-effect matrix')
    paragraph('On 39 of 43 EPA-Air windows with complete numerical history',
        r'''The executed upstream regressions reproduce each source issue in its actual pinned loader, cache builder, query collate, or RecAvg module. Changing future numerical values alters earlier released normalized chunks, while train-only normalization removes that change. Retaining no-text histories changes a fixture from three to five chunks with common numerical tensors preserved. The repaired cache origin matches raw-path note times and membership, and the repaired RecAvg axis is invariant to physical time-unit conversion. Targeted patches, code hashes, fixtures, and assertions accompany the results.''',
        'Executed upstream paths replace simplified mechanism examples')
    paragraph('The companion replay package contains original freeze records',
        r'''The additional learned grid isolates normalization, cache origin, and no-text training inclusion across all five domains and three families, plus the RecAvg-specific mixed-unit path, for 150 fits. Chronological boundaries and canonical test units remain fixed. Normalization predictions are also evaluated in common train-only coordinates, separating score scaling from learned changes. Table~\ref{tab:source-effects} reports every applicable cell. The companion package contains 1,095 new trained checkpoints, 2,175 new saved prediction sets, the existing 468-set replay, all phase protocols, mappings, token diagnostics, and execution records. Independent CUDA reloads verify the new checkpoints and fixed-model controls. Source data preserve every declared fit and test cell.''',
        'Document complete learned effects and prediction verification')
    paragraph('Finally, a hierarchical moving-block bootstrap resamples seeds',
        r'''The three-seed moving-block intervals in Figure~\ref{fig:intervals} retain the original uncertainty record. The ten-seed analysis adds paired optimization variability, while the source-effect grid quantifies implementation sensitivity in common coordinates. These analyses preserve the forecast unit and identify the component of the evaluation being varied.''',
        'Compress repeated statistical interpretation')

    paragraph('Web platforms and other data-rich systems produce numerical histories',
        r'''Web platforms produce numerical histories alongside asynchronous text, including market news, repository discussions, and operational reports. A forecaster can benefit from this channel through meaning, informative arrival times, regime-dependent frequency, entity identity, changed window membership, or added capacity. Aggregate predictive loss conflates these sources. Information-source attribution therefore requires controls that isolate the content property used to explain a gain. Related multimodal studies likewise show that task accuracy can coexist with weak cross-modal interaction or unbalanced modality use \cite{hessel2020emap,parcalabescu2023mmshap,tan2024llmuseful}.''',
        'Tighten motivation while preserving the attribution question')
    paragraph('The distinction is acute for irregular multimodal forecasting',
        r'''Time-IMM pairs numerical histories and asynchronous text across systems, networks, air quality, finance, and software repositories \cite{chang2025timeimm}. Recent methods add language representations and learned fusion \cite{lin2026timi,wang2026vot,li2026tats}. Removing text changes its occurrence process, and global permutation can change entity identity, chronology, and evaluation membership. Attribution requires changing content while preserving these evaluated objects.''',
        'Compress benchmark background')
    paragraph('The controls also separate legitimate sources of value',
        r'''Text timing and frequency can encode load, attention, incidents, or market regimes and can legitimately improve deployed forecasts when available at the cutoff. SAG asks whether the original content--time pairing contributes. CBTG asks whether content contributes beyond event positions. Their separate contrasts keep occurrence-driven value and alignment-dependent content use distinguishable.''',
        'Condense repeated interpretation')
    paragraph('The reusable audit checklist is:',
        r'''The reusable checklist is to (1) declare information available at each cutoff, (2) freeze entities, splits, event metadata, masks, targets, and windows, (3) compare aligned, within-entity/split shuffle, time-only, and numeric-only conditions, (4) identify the retrained or fixed-model estimand, (5) report paired-seed and chronology-aware uncertainty with declared multiplicity handling, (6) retain positive/null calibration and capability-screen outcomes, and (7) release invariant tests, mappings, prediction identifiers, pinned code, and commands. Resampling respects chronology \cite{bergmeir2018cv,kunsch1989block,politis1994stationary}.''',
        'Retain all seven executable checklist elements concisely')
    paragraph('The original hierarchical moving-block bootstrap',
        r'''Figure~\ref{fig:intervals} uses 10,000 unadjusted bootstrap replicates over seeds and within-entity blocks of length 3, with lengths 1 and 5 as sensitivity checks. The ten-seed analysis resamples paired seed indices 10,000 times and applies Holm correction to 30 sign tests, treating differences within $10^{-10}$ as ties. Seeds describe optimization variability conditional on the inspected data.''',
        'Consolidate uncertainty details')
    paragraph('We calibrate the original three neural families on generated historical text streams',
        r'''We calibrate all three neural families using eight EPA-Air empirical numerical/text clocks and four numerical channels. Independent text directions rendered as frozen MiniLM sentence vectors supply signal $\lambda zc$, with $c=(1,-1,.5,-.5)$. Here $z$ sums note directions with $\ell_2$-normalized exponential weights, a one-day arrival-age decay scale, and value zero for empty histories. Targets add this signal to $0.8v_{\rm last}+0.2\bar v$ and noise of SD 0.35. Arrival delays average 0.15 day, with every note available by the forecast cutoff. Train and held-out phrase templates are disjoint. Strengths are $0$, $.05$, $.1$, $.25$, and $1$.''',
        'Shorten calibration construction without a long unbreakable formula')
    paragraph('We evaluate all 150 aligned GPT-2 checkpoints under two within-history interventions',
        r'''Two interventions evaluate all 150 aligned GPT-2 checkpoints. Temporal rotation uses chronological blocks of at most four notes. Nearest-content swapping greedily pairs the most similar remaining vectors. Both operate inside the recipient history, preserving its content multiset, event metadata, numerical inputs, masks, targets, and windows. All donors arrive by its cutoff, with zero future donors across 450 evaluations. Empty and single-note histories remain.''',
        'Condense historical-control invariants')
    paragraph('The frozen encoders are',
        r'''MiniLM and FinBERT \cite{minilm_modelcard,finbert_modelcard} use masked-mean pooling, L2 normalization, and token limits of 256 and 512. All note slots remain, with per-note truncation and model-file hashes recorded. The 270 fits cross five domains, three families, three conditions, and seeds 53/67/79. Encoder-width input projections retain internal widths, budgets, and validation selection. Independent residual-ridge checks remain separately archived.''',
        'Consolidate full-grid encoder implementation details')
    paragraph('The additional learned grid isolates normalization, cache origin',
        r'''The 150-fit grid isolates normalization, cache origin, and no-text training inclusion across all five domains and three families, plus RecAvg mixed units (Table~\ref{tab:source-effects}). Test units and chronology remain fixed, with predictions in common train-only coordinates. The package includes 1,095 new checkpoints, 2,175 new predictions, the existing 468-set replay, protocols, mappings, diagnostics, and independent CUDA reload records.''',
        'Remove repeated source-grid interpretation while retaining all artifact counts')

    # Declare wide tables beside their first substantive discussion so the
    # original ACM float rules can place them before the appendix ends.
    moved_tables = []
    for label in ['tab:ten-seed', 'tab:encoders', 'tab:source-effects']:
        candidates = [match for match in re.finditer(r'\\begin\{table\*\}.*?\\end\{table\*\}', paper, re.S)
                      if '\\label{' + label + '}' in match[0]]
        assert len(candidates) == 1, label
        block = candidates[0][0]
        paper = paper.replace(block, '', 1)
        moved_tables.append(block)
    anchor = '\\label{sec:extension}'
    paper = paper.replace(anchor, anchor + '\n\n' + '\n\n'.join(moved_tables[:2]), 1)
    anchor = '\\label{sec:protocol}'
    paper = paper.replace(anchor, anchor + '\n\n' + moved_tables[2], 1)
    paper = paper.replace('ordinary model-comparison wins', 'baseline wins')
    paper = paper.replace('complete MiniLM/FinBERT replacements', 'complete MiniLM and FinBERT replacements')
    paper = paper.replace('dataset--family--encoder--intervention cells', 'crossed cells')
    paper = paper.replace('Zero/singleton histories are retained.', 'Empty and single-note histories are retained.')
    paper = paper.replace(r'\texttt{sentence-transformers/all-MiniLM-L6-v2}', r'\texttt{all-MiniLM-L6-v2}')
    paper = paper.replace(r'Signal strengths are $0,.05,.1,.25,1$.',
                          r'Signal strengths are $0$, $.05$, $.1$, $.25$, and $1$.')
    bibliography_colors = r'''
% Color-only marking for the two newly added model-card references.
\newcommand{\revisionbibitem}[2][]{%
  \ifrevision
    \ifnum\pdfstrcmp{#2}{minilm_modelcard}=0 \color{red!75!black}%
    \else\ifnum\pdfstrcmp{#2}{finbert_modelcard}=0 \color{red!75!black}%
    \else\color{black}\fi\fi
  \else\color{black}\fi
  \originalbibitem[#1]{#2}}
\AddToHook{env/thebibliography/begin}{%
  \let\originalbibitem\bibitem\let\bibitem\revisionbibitem}
'''
    paper = paper.replace('\\begin{document}', bibliography_colors + '\n\\begin{document}', 1)
    # Keep color whatsits present in both builds. Conditional omission can
    # otherwise change ACM list glue despite identical line breaking.
    paper = paper.replace(r'\ifrevision\color{red!75!black}\fi',
                          r'\ifrevision\color{red!75!black}\else\color{black}\fi')

    for suffix in ['pdf', 'svg']:
        shutil.copy2(ROOT / 'figures' / ('fig_neural_calibration.' + suffix),
                     DELIVERY / 'source/figures' / ('fig_neural_calibration.' + suffix))
    bibliography = DELIVERY / 'source/references.bib'
    entries = r'''
@misc{minilm_modelcard,
  author = {{Sentence Transformers}},
  title = {{all-MiniLM-L6-v2} Model Card},
  howpublished = {Hugging Face},
  url = {https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2},
  note = {Accessed November 5, 2026}
}
@misc{finbert_modelcard,
  author = {{Prosus AI}},
  title = {{FinBERT} Model Card},
  howpublished = {Hugging Face},
  url = {https://huggingface.co/ProsusAI/finbert},
  note = {Accessed November 5, 2026}
}
'''
    base_bibliography = re.sub(r'@misc\{(?:minilm_modelcard|finbert_modelcard),.*?\n\}',
                              '\n', bibliography.read_text(encoding='utf8'), flags=re.S)
    bibliography.write_text(base_bibliography.rstrip() + '\n' + entries, encoding='utf8')
    path.write_text(paper, encoding='utf8')
    (DELIVERY / 'qa/revision_history.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf8')
    baseline = (DELIVERY / 'qa/submitted_baseline.tex').read_text(encoding='utf8')
    original_blocks = baseline.split('\n\n')
    final_blocks = paper.split('\n\n')
    direct_changes = []
    matcher = difflib.SequenceMatcher(a=original_blocks, b=final_blocks, autojunk=False)
    for operation, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if operation == 'equal':
            continue
        preceding = '\n\n'.join(final_blocks[:new_start])
        headings = re.findall(r'\\(?:section\*?|subsection)\{([^\n]+)', preceding)
        context = headings[-1].replace('\\revised{', '').rstrip('}') if headings else '模板、作者与摘要'
        action = {'replace': '改写', 'insert': '新增', 'delete': '删除'}[operation]
        direct_changes.append(dict(reason=action + ' · ' + context,
            before='\n\n'.join(original_blocks[old_start:old_end]),
            after='\n\n'.join(final_blocks[new_start:new_end])))
    (DELIVERY / 'qa/change_blocks.json').write_text(json.dumps(direct_changes, ensure_ascii=False, indent=2), encoding='utf8')
    (DELIVERY / 'qa/source_changes.diff').write_text(''.join(difflib.unified_diff(
        baseline.splitlines(True), paper.splitlines(True), fromfile='accepted_submission.tex', tofile='expanded_camera_ready.tex')), encoding='utf8')
    print(json.dumps(dict(encoder_support=supports, added_training_fits=1095,
                          historical_evaluations=450, marked_source=str(path)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
