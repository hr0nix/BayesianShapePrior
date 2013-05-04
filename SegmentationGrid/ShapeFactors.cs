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

        public static Bernoulli LabelAverageConditional(Vector point, VectorGaussianWishart shapeParams)
        {
            VectorGaussianWishart shapeParamsTimesFactor = DistributionTimesFactor(point, shapeParams);
            return new Bernoulli(Math.Exp(shapeParamsTimesFactor.GetLogNormalizer() - shapeParams.GetLogNormalizer()));
        }

        public static Bernoulli LabelAverageConditional(Vector point, Tuple<PositiveDefiniteMatrix, Vector> shapeParams)
        {
            return new Bernoulli(Math.Exp(-0.5 * shapeParams.Item1.QuadraticForm(point - shapeParams.Item2)));
        }

        public static VectorGaussianWishart ShapeParamsAverageConditional(
            Vector point, Bernoulli label, VectorGaussianWishart shapeParams, VectorGaussianWishart result)
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
            var factorAsDistr = VectorGaussianWishart.FromNaturalParameters(0, point.Outer(point) * (-0.5), point, -0.5);
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
            Vector point, Gaussian shapeX, Gaussian shapeY, PositiveDefiniteMatrix shapeOrientation)
        {
            VectorGaussian shapeLocationTimesFactor = ShapeLocationTimesFactor(point, shapeX, shapeY, shapeOrientation);
            return new Bernoulli(Math.Exp(shapeLocationTimesFactor.GetLogNormalizer() - shapeX.GetLogNormalizer() - shapeY.GetLogNormalizer() - 0.5 * shapeOrientation.QuadraticForm(point)));
        }

        public static Bernoulli LabelAverageConditional(
            Vector point, double shapeX, double shapeY, PositiveDefiniteMatrix shapeOrientation)
        {
            Vector shapeLocation = Vector.FromArray(shapeX, shapeY);
            return new Bernoulli(Math.Exp(-0.5 * shapeOrientation.QuadraticForm(point - shapeLocation)));
        }

        public static Gaussian ShapeXAverageConditional(
            Vector point, Bernoulli label, Gaussian shapeX, Gaussian shapeY, PositiveDefiniteMatrix shapeOrientation)
        {
            return ShapeAverageConditional(point, label, shapeX, shapeY, shapeOrientation, true);
        }

        public static Gaussian ShapeYAverageConditional(
            Vector point, Bernoulli label, Gaussian shapeX, Gaussian shapeY, PositiveDefiniteMatrix shapeOrientation)
        {
            return ShapeAverageConditional(point, label, shapeX, shapeY, shapeOrientation, false);
        }

        #endregion

        #region Helpers

        private static Gaussian ShapeAverageConditional(
            Vector point, Bernoulli label, Gaussian shapeX, Gaussian shapeY, PositiveDefiniteMatrix shapeOrientation, bool resultForXCoord)
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
