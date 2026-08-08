#include "superfaiss/diversity.h"

#include "superfaiss/analytics.h" // ScoreXdPairSegmented

#include <cmath>

namespace superfaiss
{

namespace
{

// The subnormal-floor contract, reproduced file-local per this codebase's shipped
// convention. The name carries the file: novelty.cpp defines the same helper, and the
// plugin vendors every core .cpp into ONE translation unit, where two file-local helpers
// sharing a name collide (C2084) even though each compiles cleanly on its own here.
// file-local-epilogue convention (see analytics.cpp's own XdFloor, applied to every
// CrossDevice reduction): |score| < FLT_MIN -> exactly 0.0f, on every machine.
inline float XdFloorDiversityLocal(double score)
{
	const double lim = 1.1754943508222875e-38; // FLT_MIN, exactly
	if (score < lim && score > -lim)
	{
		return 0.0f;
	}
	return static_cast<float>(score);
}

// Metric::L2's shared relevance/redundancy transform (drift-and-diversity plan section
// 6.2): `f(x) = 1 - sqrt(x)/L`, strictly decreasing for every x >= 0, L > 0 -- reverses raw
// squared-distance order into a similarity-shaped scale. Retained for DISPLAY values only
// (outRelevance/outRedundancy, computed once per step for the winning candidate, below).
// `f` is exact over the reals but only non-strictly decreasing once its output is rounded
// to float32, so the argmax comparison below never materializes this transform -- see
// L2RankingRatio.
inline double L2RelevanceShapeTransform(double raw, double l2Scale)
{
	return 1.0 - std::sqrt(raw) / l2Scale;
}

// The fourth-adversarial-strike remedy (drift-and-diversity plan section 6.2, construction
// B, D-SLM1778). `u = sqrt(x)/L`, the pre-transform ratio `f(x) = 1 - u` is built from.
// Comparing `u` directly (double, never rounded to float32) removes the argmax's
// compression by construction rather than shrinking it: two distinct float32 raw distances
// `x_1 < x_2` promoted to double satisfy `sqrt(x_1)/L <= sqrt(x_2)/L` (double,
// correctly-rounded, order-preserving IEEE754 operations), with equality only where the two
// are indistinguishable at double's own 52-bit mantissa -- a gap the shipped
// int8-quantized score domain does not reach except where x_1 == x_2 already. Widening the
// COMPARISON to double while still forming `1 - u` (the rejected alternative, plan section
// 6.2) is the identical representational compression at a smaller floor, not its removal;
// this function is never composed with a `1 -` before the argmax uses it.
inline double L2RankingRatio(double raw, double l2Scale)
{
	return std::sqrt(raw) / l2Scale;
}

} // namespace

Status SelectDiverseMMR(
	const Hit* candidates, const XdQuery* candidateQueries, int32_t candidateCount,
	int32_t paddedDims, Metric metric, float lambda, int32_t k,
	const QuerySegment* segments, int32_t segmentCount, float l2Scale,
	int32_t* outSelectedIndices, float* outRelevance, float* outRedundancy)
{
	// Metric::Cosine's own recovery constant (section 6.2, D-INSP-57): the segment list's
	// own live weight sum, computed once per call -- the same list on every redundancy call
	// for this selection, not recomputed per candidate or per step. `1.0` at the degenerate
	// channelless case (segmentCount == 0 or segments == nullptr), matching
	// ScoreXdPairSegmented's own degenerate-identity convention.
	double cosineWeightSum = 1.0;
	if (metric == Metric::Cosine && segmentCount > 0 && segments != nullptr)
	{
		cosineWeightSum = 0.0;
		for (int32_t s = 0; s < segmentCount; ++s)
		{
			cosineWeightSum += static_cast<double>(segments[s].weight);
		}
	}

	const double l2ScaleD = static_cast<double>(l2Scale);
	const double lambdaD = static_cast<double>(lambda);

	for (int32_t step = 0; step < k; ++step)
	{
		int32_t bestPos = -1;
		int32_t bestIndex = 0;

		// Metric::L2 path (construction B, D-SLM1778): the argmax compares in the
		// pre-transform u-domain, double precision, never rounded to float32.
		// `bestURel`/`bestMeanURed` are the winning candidate's own ratios, saved here so
		// the display values are computed once after the loop, from the same numbers the
		// comparison used -- never recomputed per candidate.
		double bestRankKeyL2 = 0.0;
		double bestURel = 0.0;
		double bestMeanURed = 0.0;

		// Metric::Dot/Metric::Cosine path: unchanged from the shipped kernel. Neither
		// metric applies a transform to relevance (no float32 compression risk exists on
		// either -- confirmed, fourth adversarial strike, Control A), so the comparison
		// stays exactly as shipped.
		float bestScore = 0.0f;
		float bestRelevance = 0.0f;
		float bestRedundancy = 0.0f;

		for (int32_t pos = 0; pos < candidateCount; ++pos)
		{
			bool alreadySelected = false;
			for (int32_t s = 0; s < step; ++s)
			{
				if (outSelectedIndices[s] == pos)
				{
					alreadySelected = true;
					break;
				}
			}
			if (alreadySelected)
			{
				continue;
			}

			const int32_t index = candidates[pos].index;

			if (metric == Metric::L2)
			{
				// The pre-transform ratio, relevance's operand.
				const double uRel =
					L2RankingRatio(static_cast<double>(candidates[pos].score), l2ScaleD);

				// Mean of the pairwise pre-transform ratios over already-selected members,
				// in selection order (step 0's empty set is exactly 0, never a reduction
				// over zero terms -- the spec's own first-selection convention). Still
				// calls ScoreXdPairSegmented at every step regardless of lambda, so a
				// non-Ok status propagates even at lambda == 1.0 (D-SLM1782 -- redundancy
				// carries weight zero there, but the call is unconditional).
				double meanURed = 0.0;
				if (step > 0)
				{
					double acc = 0.0;
					for (int32_t s = 0; s < step; ++s)
					{
						float raw = 0.0f;
						const Status st = ScoreXdPairSegmented(candidateQueries[pos],
							candidateQueries[outSelectedIndices[s]], paddedDims, metric,
							segments, segmentCount, &raw);
						if (st != Status::Ok)
						{
							return st;
						}
						acc += L2RankingRatio(static_cast<double>(raw), l2ScaleD);
					}
					meanURed = acc / static_cast<double>(step);
				}

				// rankKey(pos) = (1 - lambda) * mean_u_red(pos) - lambda * u_rel(pos)
				// [maximize]. Algebraically equal to
				// `lambda*relevance - (1-lambda)*redundancy` (relevance = 1 - u_rel,
				// redundancy = 1 - mean_u_red) up to the additive constant `(2*lambda-1)`,
				// the same real number for every candidate at this step, dropped because it
				// changes no candidate's rank relative to any other (plan section 6.2's
				// remedy derivation). At lambda == 1.0f exactly, (1-lambda) == 0.0f exactly
				// (IEEE754), so rankKey(pos) = -u_rel(pos) -- exactly topk.h's own
				// Better(..., Metric::L2) order (ascending candidates[pos].score, ties on
				// ascending Hit.index).
				const double rankKey = (1.0 - lambdaD) * meanURed - lambdaD * uRel;

				const bool better = bestPos == -1 || rankKey > bestRankKeyL2 ||
					(rankKey == bestRankKeyL2 && index < bestIndex);
				if (better)
				{
					bestPos = pos;
					bestIndex = index;
					bestRankKeyL2 = rankKey;
					bestURel = uRel;
					bestMeanURed = meanURed;
				}
				continue;
			}

			// Relevance: the identity for Dot/Cosine (candidates[pos].score is already the
			// documented similarity on that metric's native scale).
			const float relevance = candidates[pos].score;

			// Mean reduction over already-selected members, in selection order.
			float redundancy = 0.0f;
			if (step > 0)
			{
				double acc = 0.0;
				for (int32_t s = 0; s < step; ++s)
				{
					float raw = 0.0f;
					const Status st = ScoreXdPairSegmented(candidateQueries[pos],
						candidateQueries[outSelectedIndices[s]], paddedDims, metric, segments,
						segmentCount, &raw);
					if (st != Status::Ok)
					{
						return st;
					}

					// The per-metric redundancy transform (section 6.2): Dot is the
					// identity (already the documented similarity, no correction needed);
					// Cosine recovers `sum(weight_s) - raw`, the degree-1-homogeneous form
					// that matches relevance's own channel-weight scaling.
					const double transformed = (metric == Metric::Dot)
						? static_cast<double>(raw)
						: (cosineWeightSum - static_cast<double>(raw));
					acc += transformed;
				}
				redundancy = XdFloorDiversityLocal(acc / static_cast<double>(step));
			}

			const float score = lambda * relevance - (1.0f - lambda) * redundancy;

			// argmax, ties broken on ascending candidate row index (topk.h's Better()
			// convention), never on pool position.
			const bool better =
				bestPos == -1 || score > bestScore || (score == bestScore && index < bestIndex);
			if (better)
			{
				bestPos = pos;
				bestIndex = index;
				bestScore = score;
				bestRelevance = relevance;
				bestRedundancy = redundancy;
			}
		}

		// Display values (the numbers rendered beside each result row, section 9.5):
		// computed once, for the step's winner only, never for every candidate, and never
		// the quantity the L2 comparison above used. Floored per the subnormal-floor
		// convention, applied uniformly to BOTH display outputs for every metric
		// (D-SLM1779 -- previously only outRedundancy was floored; outRelevance passed
		// through un-floored).
		if (metric == Metric::L2)
		{
			outRelevance[step] = XdFloorDiversityLocal(1.0 - bestURel);
			outRedundancy[step] = (step > 0) ? XdFloorDiversityLocal(1.0 - bestMeanURed) : 0.0f;
		}
		else
		{
			outRelevance[step] = XdFloorDiversityLocal(static_cast<double>(bestRelevance));
			outRedundancy[step] = bestRedundancy; // already floored above, or exactly 0 at step 0
		}
		outSelectedIndices[step] = bestPos;
	}
	return Status::Ok;
}

} // namespace superfaiss
