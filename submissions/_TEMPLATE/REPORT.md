# Report — MODEL — YOUR NAME (id: YOUR-ID)

Update this as you go. Short prose per section, not bullet soup: we read this, and it is where you
tell us things the JSON cannot.

---

## A1 — environment

How you built it. Anything that would not install cleanly and what you did instead. Which GPU you
have, and whether you share it.

## A2 — reproducing the published numbers

The published number you compared against, **with the table or section it comes from**, your number,
and the relative gap. If the gap is outside 10 %, say what you think is responsible and stop here —
that is a result, not a failure.

## A3 — smoke train, and what the real runs will cost

What subset you trained on, how long one epoch took, and **your extrapolated cost for checkpoint B
and for each variant in C**. Say what you extrapolated from. We approve the expensive runs from this
number, so a careless guess here means either a blocked run or a wasted one.

## B — retrain on the released training data

Final numbers, wall clock, GPU hours. How the retrained model compares to the released checkpoint
you scored in A2 — these are usually **not** identical, and that difference is itself worth
recording rather than hiding.

## C — retrain on the supplied variants

One subsection per variant id. Numbers and cost. No interpretation needed from you: just say what
you ran and what came out.

## Changes to upstream code

One line per hunk in `code/changes.diff`, saying why it was necessary. If the diff is empty, write
"none".

## Problems

Anything broken, ambiguous, undocumented, or suspicious — upstream or in our instructions. Do not
leave this empty because it looks better empty. It is the most useful section in the file.
