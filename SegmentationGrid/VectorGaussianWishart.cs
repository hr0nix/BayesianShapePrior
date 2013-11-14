using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer.Utils;

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
        public VectorGaussianWishart(
            double firstShape, PositiveDefiniteMatrix firstRate, Vector secondLocation, double secondPrecisionScale)
            : this(secondLocation.Count)
        {
            this.SetTo(firstShape, firstRate, secondLocation, secondPrecisionScale);
        }

        public VectorGaussianWishart(int dimensions)
        {
            this.LogDetMatrixCoeff = 0;
            this.MatrixCoeff = new PositiveDefiniteMatrix(dimensions, dimensions);
            this.VectorTimesMatrixCoeff = Vector.Zero(dimensions);
            this.VectorTimesMatrixTimesVectorCoeff = 0;
        }

        public double LogDetMatrixCoeff { get; private set; }

        public PositiveDefiniteMatrix MatrixCoeff { get; private set; }

        public Vector VectorTimesMatrixCoeff { get; private set; }

        public double VectorTimesMatrixTimesVectorCoeff { get; private set; }

        public int Dimension
        {
            get { return this.VectorTimesMatrixCoeff.Count; }
        }
        
        public bool IsPointMass
        {
            get { return this.Point != null ; }
        }

        public Tuple<PositiveDefiniteMatrix, Vector> Point { get; set; }

        public object Clone()
        {
            if (this.IsPointMass) throw new NotImplementedException();

            var result = new VectorGaussianWishart(this.Dimension);
            result.SetTo(this);
            return result;
        }

        public double MaxDiff(object that)
        {
            throw new NotImplementedException();
        }

        [Construction("LogDetMatrixCoeff", "MatrixCoeff", "VectorTimesMatrixCoeff", "VectorTimesMatrixTimesVectorCoeff")]
        public static VectorGaussianWishart FromNaturalParameters(
            double logDetMatrixCoeff, PositiveDefiniteMatrix matrixCoeff, Vector vectorTimesMatrixCoeff, double vectorTimesMatrixTimesVectorCoeff)
        {
            var result = new VectorGaussianWishart(vectorTimesMatrixCoeff.Count);
            
            result.LogDetMatrixCoeff = logDetMatrixCoeff;
            result.MatrixCoeff.SetTo(matrixCoeff);
            result.VectorTimesMatrixCoeff.SetTo(vectorTimesMatrixCoeff);
            result.VectorTimesMatrixTimesVectorCoeff = vectorTimesMatrixTimesVectorCoeff;

            return result;
        }

        public Wishart GetMatrixMarginal()
        {
            double firstShape, secondPrecisionScale;
            PositiveDefiniteMatrix firstRate;
            Vector secondLocation;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            return Wishart.FromShapeAndRate(firstShape, firstRate);
        }

        //public VectorGaussian GetVectorMarginalProj()
        //{
        //    // TODO: math needs checking

        //    if (this.IsVectorMarginalUniform())
        //    {
        //        return VectorGaussian.Uniform(2);
        //    }
            
        //    double firstShape, secondPrecisionScale;
        //    PositiveDefiniteMatrix firstRate;
        //    Vector secondLocation;
        //    ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

        //    PositiveDefiniteMatrix variance = (PositiveDefiniteMatrix)firstRate.Clone();
        //    variance.Scale(1.0 / ((firstShape - this.Dimension) * secondPrecisionScale));

        //    return VectorGaussian.FromMeanAndVariance(secondLocation, variance);
        //}

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
                this.LogDetMatrixCoeff == 0 &&
                this.MatrixCoeff.EqualsAll(0) &&
                this.VectorTimesMatrixCoeff.EqualsAll(0) &&
                this.VectorTimesMatrixTimesVectorCoeff == 0;
        }

        public bool IsMatrixMarginalUniform()
        {
            if (this.IsPointMass) throw new NotImplementedException();

            double firstShape, secondPrecisionScale;
            PositiveDefiniteMatrix firstRate;
            Vector secondLocation;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            return firstShape == 0.0 && firstRate.EqualsAll(0.0);
        }

        public bool IsVectorMarginalUniform()
        {
            if (this.IsPointMass) throw new NotImplementedException();

            double firstShape, secondPrecisionScale;
            PositiveDefiniteMatrix firstRate;
            Vector secondLocation;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            return (firstShape == 0.0 && firstRate.EqualsAll(0.0)) || secondPrecisionScale == 0.0;
        }

        public void SetToUniform()
        {
            if (this.IsPointMass) throw new NotImplementedException();

            this.LogDetMatrixCoeff = 0;
            this.MatrixCoeff.SetAllElementsTo(0);
            this.VectorTimesMatrixCoeff.SetAllElementsTo(0);
            this.VectorTimesMatrixTimesVectorCoeff = 0;
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
                logDetMatrix * this.LogDetMatrixCoeff +
                Matrix.TraceOfProduct(matrix, this.MatrixCoeff) +
                vectorTimesMatrix.Inner(this.VectorTimesMatrixCoeff) +
                vectorTimesMatrixTimesVector * this.VectorTimesMatrixTimesVectorCoeff -
                this.GetLogNormalizer();
        }

        public void SetTo(double firstShape, PositiveDefiniteMatrix firstRate, Vector secondLocation, double secondPrecisionScale)
        {
            Debug.Assert(firstRate.Rows == this.Dimension && firstRate.Cols == this.Dimension && secondLocation.Count == this.Dimension);
            
            if (this.IsPointMass) throw new NotImplementedException();

            this.LogDetMatrixCoeff = firstShape - this.Dimension * 0.5;
            this.MatrixCoeff.SetTo(-firstRate - secondLocation.Outer(secondLocation) * 0.5 * secondPrecisionScale);
            this.VectorTimesMatrixCoeff.SetTo(secondLocation * secondPrecisionScale);
            this.VectorTimesMatrixTimesVectorCoeff = -secondPrecisionScale * 0.5;
        }

        public void SetTo(VectorGaussianWishart value)
        {
            Debug.Assert(this.Dimension == value.Dimension);
            
            if (this.IsPointMass || value.IsPointMass) throw new NotImplementedException();

            this.LogDetMatrixCoeff = value.LogDetMatrixCoeff;
            this.MatrixCoeff.SetTo(value.MatrixCoeff);
            this.VectorTimesMatrixCoeff.SetTo(value.VectorTimesMatrixCoeff);
            this.VectorTimesMatrixTimesVectorCoeff = value.VectorTimesMatrixTimesVectorCoeff;
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

            this.LogDetMatrixCoeff = a.LogDetMatrixCoeff + b.LogDetMatrixCoeff;
            this.MatrixCoeff.SetToSum(a.MatrixCoeff, b.MatrixCoeff);
            this.VectorTimesMatrixCoeff.SetToSum(a.VectorTimesMatrixCoeff, b.VectorTimesMatrixCoeff);
            this.VectorTimesMatrixTimesVectorCoeff = a.VectorTimesMatrixTimesVectorCoeff + b.VectorTimesMatrixTimesVectorCoeff;
        }

        public void SetToRatio(VectorGaussianWishart numerator, VectorGaussianWishart denominator, bool forceProper = false)
        {
            if (this.IsPointMass || numerator.IsPointMass || denominator.IsPointMass || forceProper) throw new NotImplementedException();

            this.LogDetMatrixCoeff = numerator.LogDetMatrixCoeff - denominator.LogDetMatrixCoeff;
            this.MatrixCoeff.SetToDifference(numerator.MatrixCoeff, denominator.MatrixCoeff);
            this.VectorTimesMatrixCoeff.SetToDifference(numerator.VectorTimesMatrixCoeff, denominator.VectorTimesMatrixCoeff);
            this.VectorTimesMatrixTimesVectorCoeff = numerator.VectorTimesMatrixTimesVectorCoeff - denominator.VectorTimesMatrixTimesVectorCoeff;
        }

        public void SetToPower(VectorGaussianWishart value, double exponent)
        {
            if (this.IsPointMass || value.IsPointMass) throw new NotImplementedException();

            this.SetTo(value);
            this.LogDetMatrixCoeff *= exponent;
            this.MatrixCoeff *= exponent;
            this.VectorTimesMatrixCoeff *= exponent;
            this.VectorTimesMatrixTimesVectorCoeff *= exponent;
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

        public void ExtractParameters(
            out double firstShape, out PositiveDefiniteMatrix firstRate, out Vector secondLocation, out double secondPrecisionScale)
        {
            if (this.IsPointMass) throw new NotImplementedException();

            firstShape = this.LogDetMatrixCoeff + this.Dimension * 0.5;
            secondPrecisionScale = -2 * this.VectorTimesMatrixTimesVectorCoeff;
            secondLocation = 1.0 / secondPrecisionScale * this.VectorTimesMatrixCoeff;
            firstRate = new PositiveDefiniteMatrix(this.Dimension, this.Dimension);
            firstRate.SetTo(-this.MatrixCoeff - this.VectorTimesMatrixCoeff.Outer(secondLocation) * 0.5);
        }

        public override string ToString()
        {
            double firstShape, secondPrecisionScale;
            PositiveDefiniteMatrix firstRate;
            Vector secondLocation;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            return string.Format(
                "a={0}  B={1}  mu={2}  lambda={3}", firstShape, firstRate, secondLocation, secondPrecisionScale);
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
