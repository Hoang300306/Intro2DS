# Report — <model> — <your name / id>

Keep this updated as you go. One short section per stage. Prose, not bullet soup: we read this.

## Stage 1 — environment
How you built the environment, anything that did not install cleanly, which GPU you have.

## Stage 2 — reproducing the published numbers
The published number you are comparing against, **with a table or section reference**, your number,
and the gap. If the gap is outside 10 %, say what you think is responsible and stop here.

## Stage 3 — smoke train
What subset you trained on, how long one epoch took, and your extrapolated cost for the full run.
We budget stages 4 and 5 from this.

## Stage 4 — full retrain on the released training data
Final numbers, wall clock, GPU hours. How the retrained model compares to the released checkpoint
from stage 2 — they are usually not identical, and that difference is itself worth recording.

## Stage 5 — retrain on the supplied training-set variants
One subsection per split id. Numbers only; no interpretation needed from you.

## Changes to upstream code
One line per hunk in `code/changes.diff`, saying why. If the diff is empty, write "none".

## Problems
Anything broken, ambiguous or undocumented upstream. This section is valuable — do not leave it
empty just because it looks better that way.
