#include "superfaiss/diversity.h"

#include "superfaiss/analytics.h" // ScoreXdPair

namespace superfaiss
{

namespace
{

// The subnormal-floor contract, reproduced file-local per this codebase's shipped
// file-local-epilogue convention (see analytics.cpp's own XdFloor, applied to every
// CrossDevice reduction): |score| < FLT_MIN -> exactly 0.0f, on every machine.
inline float XdFloorLocal(double score)
{
	const double lim = 1.1754943508222875e-38; // FLT_MIN, exactly
	if (score < lim && score > -lim)
	{
		return 0.0f;
	}
	return static_cast<float>(score);
}

} // namespace

Status SelectDiverseMMR(
	const Hit* candidates, const XdQuery* candidateQueries, int32_t candidateCount,
	int32_t paddedDims, Metric metric, float lambda, int32_t k,
	int32_t* outSelectedIndices, float* outRelevance, float* outRedundancy)
{
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

			// Mean reduction over already-selected members, in selection order (step 0's
			// empty set is exactly 0, never a reduction over zero terms -- the spec's own
			// first-selection convention).
			float redundancy = 0.0f;
			if (step > 0)
			{
				double acc = 0.0;
				for (int32_t s = 0; s < step; ++s)
				{
					float pairScore = 0.0f;
					const Status st = ScoreXdPair(candidateQueries[pos],
						candidateQueries[outSelectedIndices[s]], paddedDims, metric,
						&pairScore);
					if (st != Status::Ok)
					{
						return st;
					}
					acc += static_cast<double>(pairScore);
				}
				redundancy = XdFloorLocal(acc / static_cast<double>(step));
			}

			const float relevance = candidates[pos].score;
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
