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
}
