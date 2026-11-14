"""Math on the Edge copilot: the minimum pipeline every team starts from.

Standard library only, so it runs on the UNO Q's Debian image without pip.

New in v2:
  - Smart solving strategies (few-shot, voting, calibrated)
  - Structured prompts with step-by-step reasoning
  - Confidence-based early stopping (vote_safe)

Usage:
  from copilot.pipeline import solve
  result = solve(problem, index=index, k=3)  # auto mode (best accuracy)
  
Or specify a strategy:
  result = solve(problem, index=index, k=3, strategy="vote_safe")
"""