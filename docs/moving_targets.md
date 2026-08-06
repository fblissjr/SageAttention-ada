last updated: 2026-08-06

# Working against moving targets

Most of this repo's methodology assumes a stable subject: change one
thing, measure, compare against a baseline that still means what it meant
last week. `docs/perf_research_framework.md` is written for that world and
is correct there.

A lot of the work is not in that world. When a model class is new, the
consumer nodes, the quantization tooling, and the framework's own kernels
all move within the same afternoon. On one measured day, a third-party
node this repo depends on shipped five commits, two of them
behaviour-changing at the sizes we actually render, and a sixth landed
while an experiment designed against the fifth was in flight. Upstream
README files pinned requirements that were stale in *both* directions on
the same day -- one gate had merged eleven hours earlier and was described
as still pending, another was described as available and was still an open
PR.

You cannot run controlled science against that. The mistake is to
conclude that rigour is therefore unavailable and fall back on vibes. The
correction is narrower: **rigour stops meaning "a reproducible result"
and starts meaning "a result, plus the exact state that produced it, plus
the condition under which it expires."** A number without those is not a
finding here, it is an anecdote with a decimal point.

## Rules

**Pin every dependency version with the measurement, not with the
session.** Not just ours -- torch and triton were already covered and that
turned out to be the easy half. Third-party node commits belong in the
same snapshot. The rule exists because a timing ladder and a set of audio
numbers both had to be discarded: neither recorded which build of a
fast-moving dependency produced them, and two behaviour-changing commits
had landed in the window they were taken in. The numbers were probably
fine. "Probably fine" is not a state you can defend or build on.

**Re-verify gates yourself, in both directions.** Upstream docs go stale
permissively (claiming a dependency is required when it merged hours ago)
and restrictively (claiming something is available when the PR is still
open). Both errors were observed on the same page on the same day. Check
the actual branch, the actual installed module, the actual merge state.
This costs a minute and it is the difference between a blocked afternoon
and a wasted one.

**A name in a capability list is not a capability.** A build reported
`convrot_w4a4_linear` available, which reads as evidence that a w4a8
checkpoint might load. It is not: w4a4 and w4a8 are different
quantization schemes with different layout classes, different GEMMs and
different dequant paths. The shared rotation puts `convrot` in both names.
Confirm the symbol you actually need is present, in the module that is
actually installed -- not a plausibly related one.

**Prefer a counter to a log line, and call-time evidence to install-time
evidence.** A config parsed at install time logs happily and then fails to
apply at call time, because the state it depends on never reached the call
site. The install log prints identically in both cases. Where a dispatch
counter exists, read it: a non-zero count is proof the path executed,
which no install-time message can give you. See the evidence ladder in
`docs/perf_research_framework.md`; this is the same asymmetry one layer
out, and it is what distinguishes "the arm did not help" from "the arm
did not run".

**An upstream change landing mid-experiment is a follow-up, not an
interrupt.** Measure against the pinned version, finish, then read the new
commit as a question about what to do next. Chasing the head means never
completing a comparison. The corollary is that a mid-flight upstream
commit is a reason to check whether the *conclusion space* widened, not to
discard the run: a knob that generalizes the one you are testing turns
your binary A/B into the endpoints of something continuous, which is a
better position than it sounds.

**Record the expiry condition with the finding.** Every conclusion here
should carry what would falsify it and what would make it stale. "Verdict
holds at 16:9 only; re-check quality at other aspects, timing needs no
re-check because it follows S^2" is a usable finding. "tau 2.0 is fine" is
not, and cost a walk-back.

**A local reference copy of a dependency must track what is installed, or
it will quietly answer questions about a version nobody is running.** The
`coderef/` convention exists so a third-party source tree can be read
during verification. An independent clone of one drifted two days behind
the installed node while looking entirely current; a grep for "what does
this node do" would have described code that was not executing. Prefer a
symlink to the installed tree over a separate clone, so the reference copy
cannot disagree with reality, and treat it as read-only: `git log`,
`git show`, `git diff`, `grep`. Never `checkout`, `pull`, `reset` or
`stash` there -- those mutate a working tree the framework has loaded.
`git fetch` is safe (it writes only refs and objects, never the working
tree), which is what makes a symlinked reference copy workable. Reading a
different revision does not need a checkout: `git show <rev>:<path>`.

**Keep one narrative index per fast-moving area.** Detail fragments across
session logs, memory, and one-off docs, and then conclusions go stale
without anyone noticing. One appendable file per area, newest first, in
which superseded beliefs stay visible alongside what disproved them. The
audit trail is the product: "we thought X, then measured Y" is what stops
the third person re-deriving X.

## What this is not

Not a licence for lower standards. The opposite: when the subject moves,
undisciplined work decays to noise within days, so the bookkeeping matters
*more* than it does against a stable target, not less. The tinkering is
legitimate. Tinkering without recording the state it happened in is how
you end up with four documents confidently disagreeing about the same
number.

Also not a reason to stop shipping. Waiting for the ecosystem to settle
means arriving after the interesting problems are solved. The point is to
move fast and leave a trail that survives the target moving underneath it.
