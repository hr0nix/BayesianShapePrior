using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SegmentationGrid
{
    public static class Factors
    {
        [ParameterNames("sum", "a", "b")]
        public static Vector Plus(Vector a, Vector b)
        {
            return a + b;
        }

        [ParameterNames("any", "a")]
        public static bool AnyTrue(IList<bool> array)
        {
            return array.Any(b => b);
        }
    }

    [FactorMethod(typeof(Factors), "Plus")]
    public static class PlusOps
    {
        public static VectorGaussian SumAverageConditional(VectorGaussian a, VectorGaussian b, VectorGaussian result)
        {
            result.SetMeanAndVariance(a.GetMean() + b.GetMean(), a.GetVariance() + b.GetVariance());
            return result;
        }

        public static VectorGaussian AAverageConditional(VectorGaussian sum, VectorGaussian b, VectorGaussian result)
        {
            result.SetMeanAndVariance(sum.GetMean() - b.GetMean(), sum.GetVariance() + b.GetVariance());
            return result;
        }

        public static VectorGaussian BAverageConditional(VectorGaussian sum, VectorGaussian a, VectorGaussian result)
        {
            return AAverageConditional(sum, a, result);
        }
    }

    [FactorMethod(typeof(Factors), "AnyTrue")]
    public static class AnyTrueOps
    {
        public static Bernoulli AnyAverageConditional(IList<Bernoulli> a)
        {
            double logProbFalse = 0;
            for (int i = 0; i < a.Count; ++i)
            {
                logProbFalse += a[i].GetLogProbFalse();
            }

            return new Bernoulli(1 - Math.Exp(logProbFalse));
        }

        public static TBernoulliList AAverageConditional<TBernoulliList>(TBernoulliList a, Bernoulli any, TBernoulliList result)
            where TBernoulliList : IList<Bernoulli>
        {
            double logProbFalse = 0;
            for (int i = 0; i < a.Count; ++i)
            {
                logProbFalse += a[i].GetLogProbFalse();
            }

            double anyProbTrue = any.GetProbTrue();
            for (int i = 0; i < a.Count; ++i)
            {
                double probFalseWithoutThis = Math.Exp(logProbFalse - a[i].GetLogProbFalse());
                
                double probTrue = anyProbTrue;
                double probFalse = anyProbTrue * (1 - probFalseWithoutThis) + (1 - anyProbTrue) * probFalseWithoutThis;
                result[i] = new Bernoulli(probTrue / (probTrue + probFalse));
            }
            
            return result;
        }
    }
}
