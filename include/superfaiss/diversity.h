#pragma once

#include "types.h"
#include "kernels.h" // XdQuery

// SuperFAISS V3.4 -- diversity (drift-and-diversity plan section 6). Greedy MMR (Maximal
// Marginal Relevance) selection over an already-retrieved candidate pool: reuses
// analytics.h's ScoreXdPair pairwise cross-device scorer for the redundancy term and
// Reduce::Mean's existing double-accumulate-then-divide convention (analytics.h:25) for the
// mean reduction over already-selected members. Ties among equally scored candidates break
// on ascending Hit.index -- topk.h's Better() convention (topk.h:8-19) -- never on the
// candidate's position within the over-fetched pool, which is an artifact of retrieval
// order and not guaranteed stable.

namespace superfaiss
{

// Greedy Maximal Marginal Relevance selection over `candidateCount` already-retrieved
// candidates. `candidates[i].score` is the candidate's relevance (from the query that
// produced the pool); `candidates[i].index` is the candidate's own bank-row index, read
// only for the tie-break. `candidateQueries[i]` is the same candidate's XdQuery payload,
// paired by position with `candidates[i]`, used for pairwise cross-device scoring against
// already-selected members via ScoreXdPair. `paddedDims`/`metric` are passed through to
// ScoreXdPair unchanged.
//
// `lambda` in [0, 1] weighs relevance against redundancy: at `lambda == 1.0` every
// candidate's redundancy term is weighted zero and the selection reduces to relevance
// order (the design brief's lambda=1 identity) -- a structural property of the formula
// below, not a special-cased branch.
//
// Precondition (caller-enforced, not re-checked here): `k` already validated to
// [1, candidateCount]. The plugin's over-fetch construction is this function's sole
// production caller and the plan's chosen enforcement point for this bound;
// SelectDiverseMMR does not defend against a degenerate k or candidateCount.
//
// Selection step i (0-based, i in [0, k)) scores every not-yet-selected candidate's
// `lambda * relevance - (1 - lambda) * redundancy`, where `redundancy` is the mean (double
// accumulation over already-selected members in selection order, then one divide -- the
// same convention Reduce::Mean already uses) of that candidate's ScoreXdPair distance
// against each already-selected member. At step 0 the already-selected set is empty and
// `redundancy` is exactly 0 -- never a reduction over zero terms -- the only value
// consistent with "redundant with nothing"; it makes the first pick's score a positive
// scalar multiple of relevance at every lambda > 0, independent of lambda, matching the
// lambda=1 identity for the whole query. Ties (equal combined score) break on ascending
// `candidates[pos].index`.
//
// Writes `k` entries: `outSelectedIndices[i]` is the winning candidate's POSITION within
// `candidates`/`candidateQueries` (in [0, candidateCount), suitable for indexing back into
// either array), and `outRelevance[i]`/`outRedundancy[i]` are that step's two score
// components, in selection order -- the values the caller renders beside each result row.
//
// Determinism: reuses ScoreXdPair's already-proven cross-device-exact scoring; the only new
// nondeterminism surface is the argmax tie-break, closed by the pinned ascending-index rule
// above. Propagates the first non-Ok Status any ScoreXdPair call returns (e.g.
// ZeroNormQuery on a Cosine bank whose payload has a zero self-dot).
Status SelectDiverseMMR(
	const Hit* candidates, const XdQuery* candidateQueries, int32_t candidateCount,
	int32_t paddedDims, Metric metric, float lambda, int32_t k,
	int32_t* outSelectedIndices, float* outRelevance, float* outRedundancy);

} // namespace superfaiss
