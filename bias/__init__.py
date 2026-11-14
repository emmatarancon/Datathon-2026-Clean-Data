"""bias: letter-bias diagnosis and corrections for the Math on the Edge copilot.

Phase 1 (bias/diagnose.py) separates the three hypotheses with data:
H3 measurement artifact (FALLBACK_ANSWER=D on timeouts/errors, problems
without read options), H1 letter prior, H2 position bias. Phase 2 applies only
the corrections the numbers justify: reserve letter, logprob calibration
(bias/calibration.py), permutation debiasing (bias/permute.py), through the
DEBIAS_MODE hook in copilot/pipeline.py (bias/hook.py). Standard library and
copilot imports only, so everything runs on the board without pip.
"""
