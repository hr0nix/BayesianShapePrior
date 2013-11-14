using System;
using System.Diagnostics;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace SegmentationGrid
{
    static class MathHelpers
    {
        // Actually it doesn't require matrix to be positive semidefinite
        public static PositiveDefiniteMatrix Invert(PositiveDefiniteMatrix matrix)
        {
            Debug.Assert(matrix.Rows == 2 && matrix.Cols == 2);

            double determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
            Debug.Assert(Math.Abs(determinant) > 1e-10);

            double invDeterminant = 1.0 / determinant;
            PositiveDefiniteMatrix result = new PositiveDefiniteMatrix(2, 2);
            result[0, 0] = invDeterminant * matrix[1, 1];
            result[0, 1] = -invDeterminant * matrix[0, 1];
            result[1, 0] = -invDeterminant * matrix[1, 0];
            result[1, 1] = invDeterminant * matrix[0, 0];

            return result;
        }

        // Works even for improper VectorGaussian
        public static Gaussian GetMarginal(VectorGaussian dist, int dimension)
        {
            PositiveDefiniteMatrix variance = Invert(dist.Precision);
            Vector mean = dist.MeanTimesPrecision * variance;
            return Gaussian.FromMeanAndVariance(mean[dimension], variance[dimension, dimension]);
        }

        public static PositiveDefiniteMatrix GetMean(Wishart dist)
        {
            return Invert(dist.Rate) * dist.Shape;
        }

        public static double DistanceSqr(Vector vector1, Vector vector2)
        {
            Vector diff = vector1 - vector2;
            return diff.Inner(diff);
        }
    }
}
