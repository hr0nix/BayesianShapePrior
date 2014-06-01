using System;
using System.Diagnostics;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Maths;

namespace SegmentationGrid
{
    public static class GridFactors
    {
        public static void Potts(bool a, bool b, double penalty)
        {
            throw new InvalidOperationException("This method is not supposed to be called.");
        }
    }

    [FactorMethod(typeof(GridFactors), "Potts")]
    public class PottsOps
    {
        public static Bernoulli AAverageConditional(bool b, double penalty)
        {
            double penaltyFactor = Math.Exp(-penalty);
            double bProbTrue = b ? 1 : 0;
            double probTrue = (penaltyFactor * (1 - bProbTrue) + bProbTrue) / (penaltyFactor + 1);
            return new Bernoulli(probTrue);
        }

        public static Bernoulli AAverageConditional(Bernoulli b, double penalty)
        {
            double penaltyFactor = Math.Exp(-penalty);
            double bProbTrue = b.GetProbTrue();
            double probTrue = (penaltyFactor * (1 - bProbTrue) + bProbTrue) / (penaltyFactor + 1);
            return new Bernoulli(probTrue);
        }

        public static Bernoulli BAverageConditional(bool a, double penalty)
        {
            return AAverageConditional(a, penalty);
        }

        public static Bernoulli BAverageConditional(Bernoulli a, double penalty)
        {
            return AAverageConditional(a, penalty);
        }
    }
}