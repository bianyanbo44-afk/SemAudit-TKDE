# Added-experiment protocol

All additions are exploratory, after the original test outcomes were inspected.
Existing October extension evidence is identified as such; it is not relabelled
as a pre-submission preregistration or a new natural-data confirmation.

1. Reproduce irregular semi-synthetic positive/null controls on eight empirical
   EPA-Air event clocks: 20 independently generated worlds, four signal strengths,
   oracle and actual pinned MiniLM sentence representations, fixed-model and
   separately retrained contrasts. Independently replay saved checkpoints.
   A second declared sensitivity grid adds strengths 0.025,0.05,0.1 to the same
   generator to resolve smaller effects. It does not estimate a universal
   minimum detectable effect; report null and ungated rates as well as gated ones.
2. Extend all three original neural audit families on all five datasets from
   seeds 53,67,79 to ten seeds by adding 83,97,101,113,127,139,151. Use identical
   data, caches, training budgets, paired initializations, and early-stopping rule.
   Do not select seeds or hyperparameters using the new test outcomes. Save every
   checkpoint and per-window prediction, target and mask. Report old and new
   evidence separately and combined; no equivalence claim or seed/window
   pseudoreplication.
3. Verify actual MiniLM and FinBERT representation sensitivity evidence from the
   independent, validation-selected residual-ridge audit on FNSPID and EPA-Air.
   It is not a replacement encoder experiment in the neural families. Report
   validation capability, transfer, and entity aggregation alongside contrasts.
   Additionally refit the original selective architecture on FNSPID and EPA-Air
   using full-note MiniLM embeddings, without a latest-four truncation; use the
   original three paired seeds and all three content conditions. Only the input
   width changes from 768 to 384; no encoder or hyperparameter fine-tuning.
4. Quantify rule sensitivity at 0,0.5,1,2 percent and paired-win rules, including
   strict all-seed wins. Repeat pooled support excluding low-sample CESNET.
5. Add restricted temporal-neighborhood shuffle on the same saved aligned neural
   checkpoints. It is a distributional sensitivity check, not proof of a pure
   semantic causal intervention. Save mappings and invariant checks.
6. Expose exact pinned source excerpts and tests for all four implementation
   couplings; quantify population and scaling changes separately from learning.
7. Package source, commands, tests, per-seed and window-level derived evidence;
   exclude third-party raw datasets, model weights and private identifiers.

