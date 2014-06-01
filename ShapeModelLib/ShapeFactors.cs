using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Maths;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SegmentationGrid
{
    public static class ShapeFactors
    {
        [Stochastic]
        [ParameterNames("Label", "Point", "ShapeParamsX", "ShapeParamsY")]
        public static bool LabelFromShape(Vector point, Tuple<double, double> shapeParamsX, Tuple<double, double> shapeParamsY)
        {
            Bernoulli distr = LabelFromShapeOps.LabelAverageConditional(point, shapeParamsX, shapeParamsY);
            return Rand.Double() < distr.GetProbTrue();
        }

        [Stochastic]
        [ParameterNames("Label", "Point", "ShapeParams")]
        public static bool LabelFromShape(Vector point, Tuple<PositiveDefiniteMatrix, Vector> shapeParams)
        {
            Bernoulli distr = LabelFromShapeFullCovarianceOps.LabelAverageConditional(point, shapeParams);
            return Rand.Double() < distr.GetProbTrue();
        }

        [Stochastic]
        [ParameterNames("Label", "Point", "ShapeX", "ShapeY", "ShapeOrientation")]
        public static bool LabelFromShape(Vector point, double shapeX, double shapeY, PositiveDefiniteMatrix shapeOrientation)
        {
            Bernoulli distr = LabelFromShapeFactorizedOps.LabelAverageConditional(point, shapeX, shapeY, shapeOrientation);
            return Rand.Double() < distr.GetProbTrue();
        }

        [ParameterNames("Combination", "Position", "Orientation")]
        public static Tuple<PositiveDefiniteMatrix, Vector> Combine(Vector position, PositiveDefiniteMatrix orientation)
        {
            return Tuple.Create(orientation, position);
        }

        [ParameterNames("Combination", "PositionX", "PositionY", "Orientation")]
        public static Tuple<PositiveDefiniteMatrix, Vector> Combine(double positionX, double positionY, PositiveDefiniteMatrix orientation)
        {
            return Tuple.Create(orientation, Vector.FromArray(positionX, positionY));
        }

        [ParameterNames("Matrix", "logScaleX", "logScaleY", "angle")]
        public static PositiveDefiniteMatrix MatrixFromAngleScale(double logScaleX, double logScaleY, double angle)
        {
            if (angle < -Math.PI / 4 || angle > Math.PI / 4)
            {
                throw new ConstraintViolatedException("Bad angle.");
            }
            
            Matrix scaleInvMatrix = new Matrix(2, 2);
            scaleInvMatrix[0, 0] = Math.Exp(-logScaleX);
            scaleInvMatrix[1, 1] = Math.Exp(-logScaleY);

            Matrix rotationMatrix = new Matrix(2, 2);
            rotationMatrix[0, 0] = Math.Cos(angle);
            rotationMatrix[0, 1] = -Math.Sin(angle);
            rotationMatrix[1, 0] = Math.Sin(angle);
            rotationMatrix[1, 1] = Math.Cos(angle);

            return new PositiveDefiniteMatrix(rotationMatrix * scaleInvMatrix * scaleInvMatrix * rotationMatrix.Transpose());
        }
    }

    [FactorMethod(typeof(ShapeFactors), "MatrixFromAngleScale")]
    public static class MatrixFromAngleScaleOps
    {
        public static Wishart MatrixAverageConditional(Wishart matrix, Gaussian angle, Gaussian logScaleX, Gaussian logScaleY, Wishart result)
        {
            if (!matrix.IsPointMass)
            {
                throw new NotImplementedException();
            }
            
            Func<Matrix, double> logF = 
                p =>
                {
                    double lsx, lsy, a;
                    ExtractScaleAngle(p, out lsx, out lsy, out a);

                    return angle.GetLogProb(a) + logScaleX.GetLogProb(lsx) + logScaleY.GetLogProb(lsy);
                };

            Func<Matrix, double> trXdLogF =
                p =>
                {
                    return (p * MatrixDerivative(p, logF)).Trace();
                };

            Matrix dLogF = MatrixDerivative(matrix.Point, logF);
            Matrix dTrXdLogF = MatrixDerivative(matrix.Point, trXdLogF);
            double trXdTrXdLogF = (matrix.Point * dTrXdLogF).Trace();

            LowerTriangularMatrix cholesky = new LowerTriangularMatrix(2, 2);
            cholesky.SetToCholesky(matrix.Point);
            PositiveDefiniteMatrix inverse = matrix.Point.Inverse();
            result.SetDerivatives(cholesky, inverse, new PositiveDefiniteMatrix(dLogF), trXdTrXdLogF, forceProper: true);

            return result;
        }

        public static Gaussian AngleAverageConditional(Wishart matrix)
        {
            if (!matrix.IsPointMass)
            {
                throw new NotImplementedException();
            }

            double logScaleX, logScaleY, angle;
            ExtractScaleAngle(matrix.Point, out logScaleX, out logScaleY, out angle);

            return Gaussian.PointMass(angle);
        }

        public static Gaussian LogScaleXAverageConditional(Wishart matrix)
        {
            if (!matrix.IsPointMass)
            {
                throw new NotImplementedException();
            }

            double logScaleX, logScaleY, angle;
            ExtractScaleAngle(matrix.Point, out logScaleX, out logScaleY, out angle);

            return Gaussian.PointMass(logScaleX);
        }

        public static Gaussian LogScaleYAverageConditional(Wishart matrix)
        {
            if (!matrix.IsPointMass)
            {
                throw new NotImplementedException();
            }

            double logScaleX, logScaleY, angle;
            ExtractScaleAngle(matrix.Point, out logScaleX, out logScaleY, out angle);

            return Gaussian.PointMass(logScaleY);
        }

        private static Matrix MatrixDerivative(Matrix point, Func<Matrix, double> f, double step = 0.0001)
        {
            Matrix result = new Matrix(point.Rows, point.Cols);
            for (int row = 0; row < point.Rows; ++row)
            {
                for (int column = 0; column < point.Cols; ++column)
                {
                    Matrix pointMinusEps = (Matrix)point.Clone();
                    pointMinusEps[row, column] -= step;
                    Matrix pointMinus2Eps = (Matrix)point.Clone();
                    pointMinusEps[row, column] -= 2 * step;
                    Matrix pointPlusEps = (Matrix)point.Clone();
                    pointPlusEps[row, column] += step;
                    Matrix pointPlus2Eps = (Matrix)point.Clone();
                    pointPlus2Eps[row, column] += 2 * step;

                    result[row, column] = (-f(pointPlus2Eps) + 8 * f(pointPlusEps) - 8 * f(pointMinusEps) + f(pointMinus2Eps)) / (12 * step);
                }
            }

            return result;
        }

        private static void ExtractScaleAngle(Matrix matrix, out double logScaleX, out double logScaleY, out double angle)
        {
            double w11 = matrix[0, 0];
            double w12 = matrix[0, 1];
            double w21 = matrix[1, 0];
            double w22 = matrix[1, 1];
            double tg2a = Math.Abs(w11 - w22) < 1e-8 ? 0 : (w12 + w21) / (w11 - w22);
            
            angle = 0.5 * Math.Atan(tg2a); // angle is guaranteed to be in [-pi/4, pi/4] range

            double d1 = w11 + w22;
            double d2 = (w11 - w22) * Math.Sqrt(1 + tg2a * tg2a);
            
            logScaleX = 0.5 * (MMath.Ln2 - Math.Log(d1 + d2));
            logScaleY = 0.5 * (MMath.Ln2 - Math.Log(d1 - d2));

            Debug.Assert(!double.IsNaN(logScaleX) && !double.IsNaN(logScaleY) && !double.IsNaN(angle));
            Debug.Assert(angle > -Math.PI / 4 && angle < Math.PI / 4);
        }
    }

    // This not really an EP op!
    [FactorMethod(typeof(ShapeFactors), "Combine", typeof(Vector), typeof(PositiveDefiniteMatrix))]
    public static class CombineOps
    {
        public static VectorGaussianWishart CombinationAverageConditional(
            VectorGaussian position, Wishart orientation, VectorGaussianWishart combination, VectorGaussianWishart result)
        {            
            return Combine(position, orientation, result);
        }

        public static VectorGaussian PositionAverageConditional(
            VectorGaussian position, Wishart orientation, [SkipIfUniform] VectorGaussianWishart combination, VectorGaussian result)
        {
            return ExtractVectorPart(combination, result);
        }

        public static Wishart OrientationAverageConditional(
            VectorGaussian position, Wishart orientation, [SkipIfUniform] VectorGaussianWishart combination, Wishart result)
        {
            return ExtractMatrixPart(combination, result);
        }

        public static VectorGaussian ExtractVectorPart(VectorGaussianWishart gaussianWishart, VectorGaussian result)
        {
            if (gaussianWishart.IsVectorMarginalUniform())
            {
                return VectorGaussian.Uniform(2);
            }

            double firstShape, secondPrecisionScale;
            PositiveDefiniteMatrix firstRate;
            Vector secondLocation;
            gaussianWishart.ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            result.SetMeanAndPrecision(secondLocation, MathHelpers.Invert(firstRate) * (firstShape * secondPrecisionScale));

            return result;
        }

        public static Wishart ExtractMatrixPart(VectorGaussianWishart gaussianWishart, Wishart result)
        {
            result.SetTo(gaussianWishart.GetMatrixMarginal());
            return result;
        }

        public static VectorGaussianWishart Combine(VectorGaussian position, Wishart orientation, VectorGaussianWishart result)
        {
            if (orientation.IsUniform())
            {
                result.SetToUniform();
            }
            else if (position.IsUniform())
            {
                result.SetTo(orientation.Shape, orientation.Rate, Vector.Zero(2), 0);
            }
            else
            {
                PositiveDefiniteMatrix rateTimesPrecision = new PositiveDefiniteMatrix(2, 2);
                rateTimesPrecision.SetToProduct(orientation.Rate, position.Precision);
                double trace = MathHelpers.Invert(rateTimesPrecision).Trace();
                Vector positionMean = position.MeanTimesPrecision * MathHelpers.Invert(position.Precision);
                result.SetTo(orientation.Shape, orientation.Rate, positionMean, orientation.Dimension / (orientation.Shape * trace));
            }

            return result;
        }
    }

    // This not really an EP op!
    [FactorMethod(typeof(ShapeFactors), "Combine", typeof(double), typeof(double), typeof(PositiveDefiniteMatrix))]
    public static class CombineFactorizedPositionOps
    {
        public static VectorGaussianWishart CombinationAverageConditional(
            [Proper] Gaussian positionX, [Proper] Gaussian positionY, [Proper] Wishart orientation, VectorGaussianWishart combination, VectorGaussianWishart result)
        {
            VectorGaussian position = VectorGaussian.FromMeanAndPrecision(
                Vector.FromArray(positionX.GetMean(), positionY.GetMean()),
                new PositiveDefiniteMatrix(new[,] { { positionX.Precision, 0 }, { 0, positionY.Precision } }));
            return CombineOps.Combine(position, orientation, result);
        }

        public static Gaussian PositionXAverageConditional(
            Gaussian positionX, Gaussian positionY, Wishart orientation, [Proper] VectorGaussianWishart combination)
        {
            VectorGaussian position = new VectorGaussian(2);
            return MathHelpers.GetMarginal(CombineOps.ExtractVectorPart(combination, position), 0);
        }

        public static Gaussian PositionYAverageConditional(
            Gaussian positionX, Gaussian positionY, Wishart orientation, [Proper] VectorGaussianWishart combination)
        {
            VectorGaussian position = new VectorGaussian(2);
            return MathHelpers.GetMarginal(CombineOps.ExtractVectorPart(combination, position), 1);
        }

        public static Wishart OrientationAverageConditional(
            Gaussian positionX, Gaussian positionY, Wishart orientation, [Proper] VectorGaussianWishart combination, Wishart result)
        {
            return CombineOps.ExtractMatrixPart(combination, result);
        }
    }

    [FactorMethod(typeof(ShapeFactors), "LabelFromShape", typeof(Vector), typeof(Tuple<double, double>), typeof(Tuple<double, double>))]
    public static class LabelFromShapeOps
    {
        #region Evidence

        public static double LogAverageFactor(
            bool label, Vector point, Tuple<double, double> shapeParamsX, Tuple<double, double> shapeParamsY)
        {
            double logLabelProbTrue = GetLogLabelProbTrue(point, shapeParamsX, shapeParamsY);
            return label ? logLabelProbTrue : Math.Log(1 - Math.Exp(logLabelProbTrue));
        }

        public static double LogAverageFactor(Bernoulli label, Vector point, GaussianGamma shapeParamsX, GaussianGamma shapeParamsY)
        {
            GaussianGamma shapeParamsXDistrTimesFactor = DistributionTimesFactor(point[0], shapeParamsX);
            GaussianGamma shapeParamsYDistrTimesFactor = DistributionTimesFactor(point[1], shapeParamsY);
            double labelProbFalse = label.GetProbFalse();
            double normalizerProduct = Math.Exp(
                shapeParamsXDistrTimesFactor.GetLogNormalizer() - shapeParamsX.GetLogNormalizer() +
                shapeParamsYDistrTimesFactor.GetLogNormalizer() - shapeParamsY.GetLogNormalizer());
            double averageFactor = labelProbFalse + (1 - 2 * labelProbFalse) * normalizerProduct;
            Debug.Assert(averageFactor > 0);
            return Math.Log(averageFactor);
        }

        public static double LogEvidenceRatio(
            bool label, Vector point, Tuple<double, double> shapeParamsX, Tuple<double, double> shapeParamsY)
        {
            return LogAverageFactor(label, point, shapeParamsX, shapeParamsY);
        }

        public static double LogEvidenceRatio(
            Bernoulli label, Bernoulli to_label, Vector point, GaussianGamma shapeParamsX, GaussianGamma shapeParamsY)
        {
            return LogAverageFactor(label, point, shapeParamsX, shapeParamsY) - label.GetLogAverageOf(to_label);
        }

        #endregion

        #region EP and Gibbs

        public static Bernoulli LabelAverageConditional(Vector point, GaussianGamma shapeParamsX, GaussianGamma shapeParamsY)
        {
            double logProbTrue =
                CalcLogLabelMessageForSingleCoord(point[0], shapeParamsX) +
                CalcLogLabelMessageForSingleCoord(point[1], shapeParamsY);
            return new Bernoulli(Math.Exp(logProbTrue));
        }

        public static Bernoulli LabelAverageConditional(Vector point, Tuple<double, double> shapeParamsX, Tuple<double, double> shapeParamsY)
        {
            double probTrue = GetLabelProbTrue(point, shapeParamsX, shapeParamsY);
            return new Bernoulli(probTrue);
        }

        public static GaussianGamma ShapeParamsXAverageConditional(
            Vector point, Bernoulli label, GaussianGamma shapeParamsX, GaussianGamma shapeParamsY, GaussianGamma result)
        {
            return ShapeParamsAverageConditional(point[0], point[1], label, shapeParamsX, shapeParamsY, result);
        }

        public static GaussianGamma ShapeParamsYAverageConditional(
            Vector point, Bernoulli label, GaussianGamma shapeParamsX, GaussianGamma shapeParamsY, GaussianGamma result)
        {
            return ShapeParamsAverageConditional(point[1], point[0], label, shapeParamsY, shapeParamsX, result);
        }

        #endregion

        #region Helpers

        private static GaussianGamma DistributionTimesFactor(double point, GaussianGamma shapeParamsDistr)
        {
            GaussianGamma factorAsDistr = new GaussianGamma(Vector.FromArray(0, -0.5 * point * point, point, -0.5));
            GaussianGamma result = new GaussianGamma();
            result.SetToProduct(factorAsDistr, shapeParamsDistr);
            return result;
        }

        private static double CalcLogLabelMessageForSingleCoord(double point, GaussianGamma shapeParamsDistr)
        {
            GaussianGamma shapeParamsDistrTimesFactor = DistributionTimesFactor(point, shapeParamsDistr);
            return shapeParamsDistrTimesFactor.GetLogNormalizer() - shapeParamsDistr.GetLogNormalizer();
        }

        private static GaussianGamma ShapeParamsAverageConditional(
            double coord, double otherCoord, Bernoulli label, GaussianGamma shapeParamsDistr, GaussianGamma otherShapeParamsDistr, GaussianGamma result)
        {
            GaussianGamma shapeParamsDistrTimesFactor = DistributionTimesFactor(coord, shapeParamsDistr);
            GaussianGamma otherShapeParamsDistrTimesFactor = DistributionTimesFactor(otherCoord, otherShapeParamsDistr);
            double labelProbFalse = label.GetProbFalse();
            double weight1 = labelProbFalse;
            double weight2 = Math.Exp(
                shapeParamsDistrTimesFactor.GetLogNormalizer() - shapeParamsDistr.GetLogNormalizer() +
                otherShapeParamsDistrTimesFactor.GetLogNormalizer() - otherShapeParamsDistr.GetLogNormalizer()) *
                (1 - 2 * labelProbFalse);
            var projectionOfSum = new GaussianGamma();
            projectionOfSum.SetToSum(weight1, shapeParamsDistr, weight2, shapeParamsDistrTimesFactor);
            result.SetToRatio(projectionOfSum, shapeParamsDistr);

            return result;
        }

        private static double GetLogLabelProbTrue(Vector point, Tuple<double, double> shapeParamsX, Tuple<double, double> shapeParamsY)
        {
            double distanceX = point[0] - shapeParamsX.Item2;
            double distanceY = point[1] - shapeParamsY.Item2;
            double weightedDistanceSqr = distanceX * distanceX * shapeParamsX.Item1 + distanceY * distanceY * shapeParamsY.Item1;
            return -0.5 * weightedDistanceSqr;
        }

        private static double GetLabelProbTrue(Vector point, Tuple<double, double> shapeParamsX, Tuple<double, double> shapeParamsY)
        {
            return Math.Exp(GetLogLabelProbTrue(point, shapeParamsX, shapeParamsY));
        }

        #endregion
    }

    [FactorMethod(typeof(ShapeFactors), "LabelFromShape", typeof(Vector), typeof(Tuple<PositiveDefiniteMatrix, Vector>))]
    public static class LabelFromShapeFullCovarianceOps
    {
        #region EP and Gibbs

        public static Bernoulli LabelAverageConditional(Vector point, [Proper] VectorGaussianWishart shapeParams)
        {
            VectorGaussianWishart shapeParamsTimesFactor = DistributionTimesFactor(point, shapeParams);
            return new Bernoulli(Math.Exp(shapeParamsTimesFactor.GetLogNormalizer() - shapeParams.GetLogNormalizer()));
        }

        public static Bernoulli LabelAverageConditional(Vector point, Tuple<PositiveDefiniteMatrix, Vector> shapeParams)
        {
            return new Bernoulli(Math.Exp(-0.5 * shapeParams.Item1.QuadraticForm(point - shapeParams.Item2)));
        }

        public static VectorGaussianWishart ShapeParamsAverageConditional(
            Vector point, Bernoulli label, [Proper] VectorGaussianWishart shapeParams, VectorGaussianWishart result)
        {
            VectorGaussianWishart shapeParamsTimesFactor = DistributionTimesFactor(point, shapeParams);

            double labelProbFalse = label.GetProbFalse();
            double shapeParamsWeight = labelProbFalse;
            double shapeParamsTimesFactorWeight =
                Math.Exp(shapeParamsTimesFactor.GetLogNormalizer() - shapeParams.GetLogNormalizer()) *
                (1 - 2 * labelProbFalse);

            var projectionOfSum = new VectorGaussianWishart(2);
            projectionOfSum.SetToSum(shapeParamsWeight, shapeParams, shapeParamsTimesFactorWeight, shapeParamsTimesFactor);
            Debug.Assert(projectionOfSum.IsProper()); // TODO: remove me
            result.SetToRatio(projectionOfSum, shapeParams);

            return result;
        }

        #endregion

        #region Helpers

        private static VectorGaussianWishart DistributionTimesFactor(Vector point, VectorGaussianWishart shapeParamsDistr)
        {
            var factorAsDistr = VectorGaussianWishart.FromNaturalParameters(0, new PositiveDefiniteMatrix(point.Outer(point) * (-0.5)), point, -0.5);
            VectorGaussianWishart result = new VectorGaussianWishart(2);
            result.SetToProduct(factorAsDistr, shapeParamsDistr);
            return result;
        }

        #endregion
    }

    [FactorMethod(typeof(ShapeFactors), "LabelFromShape", typeof(Vector), typeof(double), typeof(double), typeof(PositiveDefiniteMatrix))]
    public static class LabelFromShapeFactorizedOps
    {
        #region Evidence

        public static double LogAverageFactor(
            Bernoulli label, Vector point, Gaussian shapeX, Gaussian shapeY, PositiveDefiniteMatrix shapeOrientation)
        {
            VectorGaussian shapeLocationTimesFactor = ShapeLocationTimesFactor(point, shapeX, shapeY, shapeOrientation);
            double labelProbFalse = label.GetProbFalse();
            double normalizerProduct = Math.Exp(
                shapeLocationTimesFactor.GetLogNormalizer() - 0.5 * shapeOrientation.QuadraticForm(point)
                - shapeX.GetLogNormalizer() - shapeY.GetLogNormalizer());
            double averageFactor = labelProbFalse + (1 - 2 * labelProbFalse) * normalizerProduct;
            Debug.Assert(averageFactor > 0);
            return Math.Log(averageFactor);
        }

        public static double LogEvidenceRatio(
            Bernoulli label, Bernoulli to_label, Vector point, Gaussian shapeX, Gaussian shapeY, PositiveDefiniteMatrix shapeOrientation)
        {
            return LogAverageFactor(label, point, shapeX, shapeY, shapeOrientation) - label.GetLogAverageOf(to_label);
        }

        #endregion

        #region EP and Gibbs

        public static Bernoulli LabelAverageConditional(
            Vector point, Gaussian shapeX, Gaussian shapeY, Wishart shapeOrientation)
        {
            if (shapeOrientation.IsPointMass)
            {
                if (shapeX.IsPointMass && shapeY.IsPointMass)
                {
                    return LabelAverageConditional(point, shapeX.Point, shapeY.Point, shapeOrientation.Point);
                }
                else if (!shapeX.IsPointMass && !shapeY.IsPointMass)
                {
                    VectorGaussian shapeLocationTimesFactor = ShapeLocationTimesFactor(point, shapeX, shapeY, shapeOrientation.Point);
                    return new Bernoulli(Math.Exp(shapeLocationTimesFactor.GetLogNormalizer() - shapeX.GetLogNormalizer() - shapeY.GetLogNormalizer() - 0.5 * shapeOrientation.Point.QuadraticForm(point)));
                }
                else
                {
                    throw new NotSupportedException();
                }
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public static Bernoulli LabelAverageConditional(
            Vector point, double shapeX, double shapeY, PositiveDefiniteMatrix shapeOrientation)
        {
            Vector shapeLocation = Vector.FromArray(shapeX, shapeY);
            return new Bernoulli(Math.Exp(-0.5 * shapeOrientation.QuadraticForm(point - shapeLocation)));
        }

        public static Gaussian ShapeXAverageConditional(
            Vector point, Bernoulli label, Gaussian shapeX, Gaussian shapeY, Wishart shapeOrientation)
        {
            return ShapeAverageConditional(point, label, shapeX, shapeY, shapeOrientation.Point, true);
        }

        public static Gaussian ShapeYAverageConditional(
            Vector point, Bernoulli label, Gaussian shapeX, Gaussian shapeY, Wishart shapeOrientation)
        {
            return ShapeAverageConditional(point, label, shapeX, shapeY, shapeOrientation.Point, false);
        }

        public static Wishart ShapeOrientationAverageConditional(
            Vector point, Bernoulli label, Gaussian shapeX, Gaussian shapeY, Wishart shapeOrientation, Wishart result)
        {
            if (shapeOrientation.IsPointMass && shapeX.IsPointMass && shapeY.IsPointMass)
            {
                double labelProbTrue = label.GetProbTrue();
                double labelProbFalse = 1.0 - labelProbTrue;
                double probDiff = labelProbTrue - labelProbFalse;

                Vector shapeLocation = Vector.FromArray(shapeX.Point, shapeY.Point);
                Vector diff = shapeLocation - point;
                Matrix diffOuter = diff.Outer(diff);
                Matrix orientationTimesDiffOuter = shapeOrientation.Point * diffOuter;
                double trace = orientationTimesDiffOuter.Trace();

                double factorValue = Math.Exp(-0.5 * shapeOrientation.Point.QuadraticForm(diff));
                double funcValue = factorValue * probDiff + labelProbFalse;

                PositiveDefiniteMatrix dLogFunc = new PositiveDefiniteMatrix(diffOuter * (-0.5 * probDiff * factorValue / funcValue));
                double xxddLogFunc =
                    -0.5 * probDiff * (-0.5 * labelProbFalse * factorValue * trace * trace / (funcValue * funcValue) + factorValue * trace / funcValue);
                
                LowerTriangularMatrix cholesky = new LowerTriangularMatrix(2, 2);
                cholesky.SetToCholesky(shapeOrientation.Point);
                PositiveDefiniteMatrix inverse = shapeOrientation.Point.Inverse();
                result.SetDerivatives(cholesky, inverse, dLogFunc, xxddLogFunc, forceProper: true);
                return result;
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        #endregion

        #region Helpers

        private static Gaussian ShapeAverageConditional(
            Vector point, Bernoulli label, Gaussian shapeX, Gaussian shapeY, PositiveDefiniteMatrix shapeOrientation, bool resultForXCoord)
        {
            if (shapeX.IsPointMass && shapeY.IsPointMass)
            {
                double labelProbTrue = label.GetProbTrue();
                double labelProbFalse = 1.0 - labelProbTrue;
                double probDiff = labelProbTrue - labelProbFalse;

                Vector shapeLocation = Vector.FromArray(shapeX.Point, shapeY.Point);
                Vector diff = point - shapeLocation;
                Vector orientationTimesDiff = shapeOrientation * diff;
                Matrix orientationTimesDiffOuter = orientationTimesDiff.Outer(orientationTimesDiff);

                double factorValue = Math.Exp(-0.5 * shapeOrientation.QuadraticForm(diff));
                double funcValue = factorValue * probDiff + labelProbFalse;
                
                Vector dFunc = probDiff * factorValue * orientationTimesDiff;
                Vector dLogFunc = 1.0 / funcValue * dFunc;
                Matrix ddLogFunc =
                    ((orientationTimesDiffOuter + shapeOrientation) * factorValue * funcValue - orientationTimesDiffOuter * probDiff * factorValue * factorValue)
                    * (probDiff / (funcValue * funcValue));

                double x = resultForXCoord ? shapeX.Point : shapeY.Point;
                double d = resultForXCoord ? dLogFunc[0] : dLogFunc[1];
                double dd = resultForXCoord ? ddLogFunc[0, 0] : ddLogFunc[1, 1];
                return Gaussian.FromDerivatives(x, d, dd, forceProper: true);
            }
            else if (!shapeX.IsPointMass && !shapeY.IsPointMass)
            {
                VectorGaussian shapeLocationTimesFactor = ShapeLocationTimesFactor(point, shapeX, shapeY, shapeOrientation);
                double labelProbFalse = label.GetProbFalse();
                double shapeLocationWeight = labelProbFalse;
                double shapeLocationTimesFactorWeight =
                    Math.Exp(shapeLocationTimesFactor.GetLogNormalizer() - shapeX.GetLogNormalizer() - shapeY.GetLogNormalizer() - 0.5 * shapeOrientation.QuadraticForm(point)) *
                    (1 - 2 * labelProbFalse);

                var projectionOfSum = new Gaussian();
                projectionOfSum.SetToSum(
                    shapeLocationWeight,
                    resultForXCoord ? shapeX : shapeY,
                    shapeLocationTimesFactorWeight,
                    shapeLocationTimesFactor.GetMarginal(resultForXCoord ? 0 : 1));
                Gaussian result = new Gaussian();
                result.SetToRatio(projectionOfSum, resultForXCoord ? shapeX : shapeY);

                return result;
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        private static VectorGaussian ShapeLocationTimesFactor(
            Vector point, Gaussian shapeX, Gaussian shapeY, PositiveDefiniteMatrix shapeOrientation)
        {
            VectorGaussian shapeLocationDistr = VectorGaussian.FromMeanAndVariance(
                Vector.FromArray(shapeX.GetMean(), shapeY.GetMean()),
                new PositiveDefiniteMatrix(new double[,] { { shapeX.GetVariance(), 0.0 }, { 0.0, shapeY.GetVariance() } }));
            VectorGaussian factorDistribution = VectorGaussian.FromMeanAndPrecision(point, shapeOrientation);
            VectorGaussian result = new VectorGaussian(2);
            result.SetToProduct(shapeLocationDistr, factorDistribution);
            return result;
        }

        #endregion
    }
}
