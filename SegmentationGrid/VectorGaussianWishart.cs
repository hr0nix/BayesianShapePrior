using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SegmentationGrid
{
    public class VectorGaussianWishart :
        IDistribution<Tuple<PositiveDefiniteMatrix, Vector>>,
        SettableTo<VectorGaussianWishart>,
        SettableToProduct<VectorGaussianWishart>,
        SettableToRatio<VectorGaussianWishart>,
        SettableToPower<VectorGaussianWishart>,
        SettableToWeightedSum<VectorGaussianWishart>,
        CanGetMean<Tuple<PositiveDefiniteMatrix, Vector>>,
        CanGetLogAverageOf<VectorGaussianWishart>,
        CanGetLogAverageOfPower<VectorGaussianWishart>,
        CanGetAverageLog<VectorGaussianWishart>,
        Sampleable<Tuple<PositiveDefiniteMatrix, Vector>>,
        CanGetLogNormalizer
    {
        private double logDetMatrixCoeff;

        private PositiveDefiniteMatrix matrixCoeff;

        private Vector vectorTimesMatrixCoeff;

        private double vectorTimesMatrixTimesVectorCoeff;

        public VectorGaussianWishart(
            double firstShape, PositiveDefiniteMatrix firstRate, Vector secondLocation, double secondPrecisionScale)
            : this(secondLocation.Count)
        {
            this.SetTo(firstShape, firstRate, secondLocation, secondPrecisionScale);
        }

        public VectorGaussianWishart(int dimensions)
        {
            this.logDetMatrixCoeff = 0;
            this.matrixCoeff = new PositiveDefiniteMatrix(dimensions, dimensions);
            this.vectorTimesMatrixCoeff = Vector.Zero(dimensions);
            this.vectorTimesMatrixTimesVectorCoeff = 0;
        }

        public int Dimension
        {
            get { return this.vectorTimesMatrixCoeff.Count; }
        }
        
        public bool IsPointMass
        {
            get { return this.Point != null ; }
        }

        public Tuple<PositiveDefiniteMatrix, Vector> Point { get; set; }

        public object Clone()
        {
            if (this.IsPointMass) throw new NotImplementedException();

            VectorGaussianWishart result = new VectorGaussianWishart(this.Dimension);
            result.SetTo(this);
            return result;
        }

        public double MaxDiff(object that)
        {
            throw new NotImplementedException();
        }

        public static VectorGaussianWishart FromNaturalParameters(
            double logDetMatrixCoeff, PositiveDefiniteMatrix matrixCoeff, Vector vectorTimesMatrixCoeff, double vectorTimesMatrixTimesVectorCoeff)
        {
            VectorGaussianWishart result = new VectorGaussianWishart(vectorTimesMatrixCoeff.Count);
            
            result.logDetMatrixCoeff = logDetMatrixCoeff;
            result.matrixCoeff.SetTo(matrixCoeff);
            result.vectorTimesMatrixCoeff.SetTo(vectorTimesMatrixCoeff);
            result.vectorTimesMatrixTimesVectorCoeff = vectorTimesMatrixTimesVectorCoeff;

            return result;
        }

        public bool IsProper()
        {
            if (this.IsPointMass) throw new NotImplementedException();

            double firstShape, secondPrecisionScale;
            PositiveDefiniteMatrix firstRate;
            Vector secondLocation;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            return
                firstShape > 0.5 * (this.Dimension + 1) && // TODO: firstShape > 0.5 * (this.Dimension - 1) according to Wikipedia
                firstRate.IsPositiveDefinite() &&
                secondPrecisionScale > 0;
        }

        public bool IsUniform()
        {
            if (this.IsPointMass) throw new NotImplementedException();

            return
                this.logDetMatrixCoeff == 0 &&
                this.matrixCoeff.EqualsAll(0) &&
                this.vectorTimesMatrixCoeff.EqualsAll(0) &&
                this.vectorTimesMatrixTimesVectorCoeff == 0;
        }

        public void SetToUniform()
        {
            if (this.IsPointMass) throw new NotImplementedException();

            this.logDetMatrixCoeff = 0;
            this.matrixCoeff.SetAllElementsTo(0);
            this.vectorTimesMatrixCoeff.SetAllElementsTo(0);
            this.vectorTimesMatrixTimesVectorCoeff = 0;
        }

        public Tuple<PositiveDefiniteMatrix, Vector> GetMean()
        {
            if (this.IsPointMass) throw new NotImplementedException();
            if (!this.IsProper()) throw new ImproperDistributionException(this);
            
            double firstShape, secondPrecisionScale;
            PositiveDefiniteMatrix firstRate;
            Vector secondLocation;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            PositiveDefiniteMatrix firstScale = new PositiveDefiniteMatrix(this.Dimension, this.Dimension);
            firstScale.SetToInverse(firstRate);
            return Tuple.Create(firstScale * firstShape, secondLocation);
        }

        public double GetLogProb(Tuple<PositiveDefiniteMatrix, Vector> value)
        {
            double logDetMatrix, vectorTimesMatrixTimesVector;
            PositiveDefiniteMatrix matrix;
            Vector vectorTimesMatrix;
            this.ExtractNaturalStatistics(value, out logDetMatrix, out matrix, out vectorTimesMatrix, out vectorTimesMatrixTimesVector);
            
            return
                logDetMatrix * this.logDetMatrixCoeff +
                Matrix.TraceOfProduct(matrix, this.matrixCoeff) +
                vectorTimesMatrix.Inner(this.vectorTimesMatrixCoeff) +
                vectorTimesMatrixTimesVector * this.vectorTimesMatrixTimesVectorCoeff -
                this.GetLogNormalizer();
        }

        public void SetTo(double firstShape, PositiveDefiniteMatrix firstRate, Vector secondLocation, double secondPrecisionScale)
        {
            Debug.Assert(firstRate.Rows == this.Dimension && firstRate.Cols == this.Dimension && secondLocation.Count == this.Dimension);
            
            if (this.IsPointMass) throw new NotImplementedException();

            this.logDetMatrixCoeff = firstShape - this.Dimension * 0.5;
            this.matrixCoeff.SetTo(-firstRate - secondLocation.Outer(secondLocation) * 0.5 * secondPrecisionScale);
            this.vectorTimesMatrixCoeff.SetTo(secondLocation * secondPrecisionScale);
            this.vectorTimesMatrixTimesVectorCoeff = -secondPrecisionScale * 0.5;
        }

        public void SetTo(VectorGaussianWishart value)
        {
            Debug.Assert(this.Dimension == value.Dimension);
            
            if (this.IsPointMass || value.IsPointMass) throw new NotImplementedException();

            this.logDetMatrixCoeff = value.logDetMatrixCoeff;
            this.matrixCoeff.SetTo(value.matrixCoeff);
            this.vectorTimesMatrixCoeff.SetTo(value.vectorTimesMatrixCoeff);
            this.vectorTimesMatrixTimesVectorCoeff = value.vectorTimesMatrixTimesVectorCoeff;
        }

        public void SetToSum(double weight1, VectorGaussianWishart value1, double weight2, VectorGaussianWishart value2)
        {
            if (this.IsPointMass || value1.IsPointMass || value2.IsPointMass) throw new NotImplementedException();

            double weightSum = weight1 + weight2;
            weight1 /= weightSum;
            weight2 /= weightSum;

            double meanLogDetMatrix1, meanLogDetMatrix2, meanVectorTimesMatrixTimesVector1, meanVectorTimesMatrixTimesVector2;
            PositiveDefiniteMatrix meanMatrix1, meanMatrix2;
            Vector meanVectorTimesMatrix1, meanVectorTimesMatrix2;

            value1.ComputeNaturalStatisticMoments(
                out meanLogDetMatrix1, out meanMatrix1, out meanVectorTimesMatrix1, out meanVectorTimesMatrixTimesVector1);
            value2.ComputeNaturalStatisticMoments(
                out meanLogDetMatrix2, out meanMatrix2, out meanVectorTimesMatrix2, out meanVectorTimesMatrixTimesVector2);

            double meanLogDetMatrixToMatch = weight1 * meanLogDetMatrix1 + weight2 * meanLogDetMatrix2;
            PositiveDefiniteMatrix meanMatrixToMatch = meanMatrix1 * weight1 + meanMatrix2 * weight2;
            Vector meanVectorTimesMatrixToMatch = weight1 * meanVectorTimesMatrix1 + weight2 * meanVectorTimesMatrix2;
            double meanVectorTimesMatrixTimesVectorToMatch = weight1 * meanVectorTimesMatrixTimesVector1 + weight2 * meanVectorTimesMatrixTimesVector2;

            Wishart distributionOverFirst = Wishart.FromMeanAndMeanLogDeterminant(meanMatrixToMatch, meanLogDetMatrixToMatch);
            Vector secondLocation = distributionOverFirst.Rate * meanVectorTimesMatrixToMatch * (1.0 / distributionOverFirst.Shape);
            double secondPrecisionScale =
                this.Dimension / (meanVectorTimesMatrixTimesVectorToMatch - secondLocation.Inner(meanVectorTimesMatrixToMatch));

            this.SetTo(distributionOverFirst.Shape, distributionOverFirst.Rate, secondLocation, secondPrecisionScale);
        }

        public void SetToProduct(VectorGaussianWishart a, VectorGaussianWishart b)
        {
            if (this.IsPointMass || a.IsPointMass || b.IsPointMass) throw new NotImplementedException();

            this.logDetMatrixCoeff = a.logDetMatrixCoeff + b.logDetMatrixCoeff;
            this.matrixCoeff.SetToSum(a.matrixCoeff, b.matrixCoeff);
            this.vectorTimesMatrixCoeff.SetToSum(a.vectorTimesMatrixCoeff, b.vectorTimesMatrixCoeff);
            this.vectorTimesMatrixTimesVectorCoeff = a.vectorTimesMatrixTimesVectorCoeff + b.vectorTimesMatrixTimesVectorCoeff;
        }

        public void SetToRatio(VectorGaussianWishart numerator, VectorGaussianWishart denominator, bool forceProper = false)
        {
            if (this.IsPointMass || numerator.IsPointMass || denominator.IsPointMass || forceProper) throw new NotImplementedException();

            this.logDetMatrixCoeff = numerator.logDetMatrixCoeff - denominator.logDetMatrixCoeff;
            this.matrixCoeff.SetToDifference(numerator.matrixCoeff, denominator.matrixCoeff);
            this.vectorTimesMatrixCoeff.SetToDifference(numerator.vectorTimesMatrixCoeff, denominator.vectorTimesMatrixCoeff);
            this.vectorTimesMatrixTimesVectorCoeff = numerator.vectorTimesMatrixTimesVectorCoeff - denominator.vectorTimesMatrixTimesVectorCoeff;
        }

        public void SetToPower(VectorGaussianWishart value, double exponent)
        {
            if (this.IsPointMass || value.IsPointMass) throw new NotImplementedException();

            this.SetTo(value);
            this.logDetMatrixCoeff *= exponent;
            this.matrixCoeff *= exponent;
            this.vectorTimesMatrixCoeff *= exponent;
            this.vectorTimesMatrixTimesVectorCoeff *= exponent;
        }

        public double GetLogAverageOf(VectorGaussianWishart that)
        {
            throw new NotImplementedException();
        }

        public double GetLogAverageOfPower(VectorGaussianWishart that, double power)
        {
            throw new NotImplementedException();
        }

        public double GetAverageLog(VectorGaussianWishart that)
        {
            throw new NotImplementedException();
        }

        public double GetLogNormalizer()
        {
            if (this.IsPointMass) throw new NotImplementedException();
            if (!this.IsProper()) throw new ImproperDistributionException(this);

            double firstShape, secondPrecisionScale;
            PositiveDefiniteMatrix firstRate;
            Vector secondLocation;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);
            double result =
                - firstShape * firstRate.LogDeterminant()
                - this.Dimension * 0.5 * Math.Log(secondPrecisionScale)
                + this.Dimension * MMath.LnSqrt2PI
                + MMath.GammaLn(firstShape, this.Dimension);
            Debug.Assert(!double.IsNaN(result));

            return result;
        }

        public Tuple<PositiveDefiniteMatrix, Vector> Sample(Tuple<PositiveDefiniteMatrix, Vector> result)
        {
            throw new NotImplementedException();
        }

        public Tuple<PositiveDefiniteMatrix, Vector> Sample()
        {
            throw new NotImplementedException();
        }

        private void ExtractParameters(
            out double firstShape, out PositiveDefiniteMatrix firstRate, out Vector secondLocation, out double secondPrecisionScale)
        {
            if (this.IsPointMass) throw new NotImplementedException();

            firstShape = this.logDetMatrixCoeff + this.Dimension * 0.5;
            secondPrecisionScale = -2 * this.vectorTimesMatrixTimesVectorCoeff;
            secondLocation = 1.0 / secondPrecisionScale * this.vectorTimesMatrixCoeff;
            firstRate = new PositiveDefiniteMatrix(this.Dimension, this.Dimension);
            firstRate.SetTo(-this.matrixCoeff - this.vectorTimesMatrixCoeff.Outer(secondLocation) * 0.5);
        }

        private void ComputeNaturalStatisticMoments(
            out double meanLogDetMatrix,
            out PositiveDefiniteMatrix meanMatrix,
            out Vector meanVectorTimesMatrix,
            out double meanVectorTimesMatrixTimesVector)
        {
            if (this.IsPointMass) throw new NotImplementedException();

            double firstShape, secondPrecisionScale;
            PositiveDefiniteMatrix firstRate;
            Vector secondLocation;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            Wishart firstDistribution = Wishart.FromShapeAndRate(firstShape, firstRate);

            meanLogDetMatrix = firstDistribution.GetMeanLogDeterminant();
            meanMatrix = firstDistribution.GetMean();
            meanVectorTimesMatrix = meanMatrix * secondLocation;
            meanVectorTimesMatrixTimesVector = this.Dimension / secondPrecisionScale + secondLocation.Inner(meanVectorTimesMatrix);
        }

        private void ExtractNaturalStatistics(
            Tuple<PositiveDefiniteMatrix, Vector> value,
            out double logDetMatrix,
            out PositiveDefiniteMatrix matrix,
            out Vector vectorTimesMatrix,
            out double vectorTimesMatrixTimesVector)
        {
            if (this.IsPointMass) throw new NotImplementedException();

            matrix = value.Item1;
            Vector vector = value.Item2;
            logDetMatrix = matrix.LogDeterminant();
            vectorTimesMatrix = vector * matrix;
            vectorTimesMatrixTimesVector = matrix.QuadraticForm(vector);
        }
    }
}
