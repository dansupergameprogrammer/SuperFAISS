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
// squared-distance order into a similarity-shaped scale, applied identically to
// candidates[pos].score (relevance) and to the candidate-to-selected pairwise distance
// (redundancy).
inline double L2RelevanceShapeTransform(double raw, double l2Scale)
{
	return 1.0 - std::sqrt(raw) / l2Scale;
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

	for (int32_t step = 0; step < k; ++step)
	{
		int32_t bestPos = -1;
		int32_t bestIndex = 0;
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

			// Relevance: the identity for Dot/Cosine (candidates[pos].score is already the
			// documented similarity on that metric's native scale); L2's shared f(x)
			// transform applied to the raw squared distance (section 6.2's L2 bullet).
			float relevance = candidates[pos].score;
			if (metric == Metric::L2)
			{
				relevance = static_cast<float>(L2RelevanceShapeTransform(
					static_cast<double>(candidates[pos].score), static_cast<double>(l2Scale)));
			}

			// Mean reduction over already-selected members, in selection order (step 0's
			// empty set is exactly 0, never a reduction over zero terms -- the spec's own
			// first-selection convention).
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
					// that matches relevance's own channel-weight scaling; L2 applies the
					// identical f(x) relevance itself already used, to this pairwise
					// distance instead of the query-to-candidate one.
					double transformed;
					if (metric == Metric::Dot)
					{
						transformed = static_cast<double>(raw);
					}
					else if (metric == Metric::Cosine)
					{
						transformed = cosineWeightSum - static_cast<double>(raw);
					}
					else // Metric::L2
					{
						transformed = L2RelevanceShapeTransform(static_cast<double>(raw),
							static_cast<double>(l2Scale));
					}
					acc += transformed;
				}
				redundancy = XdFloorDiversityLocal(acc / static_cast<double>(step));
			}

			const float score = lambda * relevance - (1.0f - lambda) * redundancy;
			const int32_t index = candidates[pos].index;

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

		outSelectedIndices[step] = bestPos;
		outRelevance[step] = bestRelevance;
		outRedundancy[step] = bestRedundancy;
	}
	return Status::Ok;
}

} // namespace superfaiss
