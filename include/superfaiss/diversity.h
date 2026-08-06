#pragma once

#include "types.h"
#include "kernels.h" // XdQuery

// SuperFAISS V3.4 -- diversity (drift-and-diversity plan section 6, Gate 0b rebuild).
// Greedy MMR (Maximal Marginal Relevance) selection over an already-retrieved candidate
// pool: reuses analytics.h's ScoreXdPairSegmented pairwise, segmented, weighted
// cross-device scorer for the redundancy term (its channelless/default-weight degenerate
// case is bit-identical to ScoreXdPair, which this rebuild replaces as the direct call) and
// Reduce::Mean's existing double-accumulate-then-divide convention (analytics.h:25) for the
// mean reduction over already-selected members. Ties among equally scored candidates break
// on ascending Hit.index -- topk.h's Better() convention (topk.h:8-19) -- never on the
// candidate's position within the over-fetched pool, which is an artifact of retrieval
// order and not guaranteed stable.

namespace superfaiss
{

// Greedy Maximal Marginal Relevance selection over `candidateCount` already-retrieved
// candidates. `candidates[i].score` is the candidate's RAW relevance (from the query that
// produced the pool, in the metric's own native scale -- untransformed for Dot/Cosine, the
// raw squared distance for L2); `candidates[i].index` is the candidate's own bank-row index,
// read only for the tie-break. `candidateQueries[i]` is the same candidate's XdQuery
// payload, paired by position with `candidates[i]`, used for pairwise cross-device scoring
// against already-selected members via ScoreXdPairSegmented. `paddedDims`/`metric` are
// passed through to ScoreXdPairSegmented unchanged.
//
// `segments`/`segmentCount` -- the caller's already-resolved channel-weight segment list,
// threaded to every redundancy call via ScoreXdPairSegmented; `segmentCount == 0` (or
// `segments == nullptr`) is the degenerate channelless case, bit-identical to the
// pre-primitive ScoreXdPair-based composition. `l2Scale` -- the bank-intrinsic scale `L`
// (`L = sqrt(Spread(current))`, the caller's own whole-row SpreadCrossDevice call, computed
// and cached by the caller: this function has no BankView and cannot call SpreadCrossDevice
// itself). Meaningful only when `metric == Metric::L2`; ignored for `Metric::Dot` and
// `Metric::Cosine`, which carry no bank-intrinsic scale of their own. The caller passes
// `0.0f` for `l2Scale` on every non-L2 metric -- a placeholder value this function never
// reads on those paths.
//
// `lambda` in [0, 1] weighs relevance against redundancy: at `lambda == 1.0` every
// candidate's redundancy term is weighted zero and the selection reduces to relevance
// order (the design brief's lambda=1 identity) -- a structural property of the formula
// below, not a special-cased branch.
//
// Precondition (caller-enforced, not re-checked here): `k` already validated to
// [1, candidateCount]. The plugin's over-fetch construction is this function's sole
// production caller and the plan's chosen enforcement point for this bound;
// SelectDiverseMMR does not defend against a degenerate k or candidateCount. Likewise,
// when `metric == Metric::L2`, the caller is required to pass `l2Scale > 0` -- the
// widget's own Metric::L2 compute path is the plan's chosen enforcement point for this
// precondition too; not re-checked here.
//
// The redundancy term, per metric (drift-and-diversity plan section 6.2, D-INSP-47/57/50/53):
// each selection step scores every not-yet-selected candidate's
// `lambda * relevance - (1 - lambda) * redundancy`, where `relevance` is `candidates[pos]
// .score` for Dot/Cosine and `1 - sqrt(candidates[pos].score) / l2Scale` for L2 (the same
// monotone-decreasing recovery applied to the redundancy operand below, so both are the
// identical transform of the same raw kernel output), and `redundancy` is the mean (double
// accumulation over already-selected members in selection order, then one divide -- the
// same convention Reduce::Mean already uses) of that candidate's TRANSFORMED distance
// against each already-selected member:
//   - Dot:    the untransformed ScoreXdPairSegmented output (already the documented
//             similarity, on the same scale as candidates[pos].score).
//   - Cosine: `sum(weight_s) - ScoreXdPairSegmented(...)`, where `sum(weight_s)` is the
//             segment list's own live weight sum (`1.0` at segmentCount == 0) -- the
//             degree-1-homogeneous recovery (D-INSP-57) that matches relevance's own
//             channel-weight scaling at every weight vector, not a fixed `1 -`.
//   - L2:     `1 - sqrt(ScoreXdPairSegmented(...)) / l2Scale` -- the identical transform
//             applied to relevance, applied here to the candidate-to-selected pairwise
//             distance instead of the query-to-candidate one.
// At step 0 the already-selected set is empty and `redundancy` is exactly 0 -- never a
// reduction over zero terms -- the only value consistent with "redundant with nothing"; it
// makes the first pick's score a positive scalar multiple of relevance at every
// lambda > 0, independent of lambda and of metric, matching the lambda=1 identity for the
// whole query. Ties (equal combined score) break on ascending `candidates[pos].index`.
//
// Writes `k` entries: `outSelectedIndices[i]` is the winning candidate's POSITION within
// `candidates`/`candidateQueries` (in [0, candidateCount), suitable for indexing back into
// either array), and `outRelevance[i]`/`outRedundancy[i]` are that step's two, mutually
// scale-consistent score components, in selection order -- the values the caller renders
// beside each result row.
//
// Determinism: reuses ScoreXdPairSegmented's already-proven cross-device-exact scoring; the
// only new nondeterminism surface is the argmax tie-break, closed by the pinned
// ascending-index rule above. Propagates the first non-Ok Status any ScoreXdPairSegmented
// call returns (e.g. ZeroNormQuery on a Cosine bank whose payload has a zero aggregate
// weighted self-norm).
Status SelectDiverseMMR(
	const Hit* candidates, const XdQuery* candidateQueries, int32_t candidateCount,
	int32_t paddedDims, Metric metric, float lambda, int32_t k,
	const QuerySegment* segments, int32_t segmentCount, float l2Scale,
	int32_t* outSelectedIndices, float* outRelevance, float* outRedundancy);

} // namespace superfaiss
