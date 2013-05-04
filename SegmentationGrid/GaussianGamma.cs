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
    public class GaussianGamma :
        IDistribution<Tuple<double, double>>,
        SettableTo<GaussianGamma>,
        SettableToProduct<GaussianGamma>,
        SettableToRatio<GaussianGamma>,
        SettableToPower<GaussianGamma>,
        SettableToWeightedSum<GaussianGamma>,
        CanGetMean<Tuple<double, double>>,
        CanGetVariance<Tuple<double, double>>,
        CanSetMeanAndVariance<Tuple<double, double>, Tuple<double, double>>,
        CanGetLogAverageOf<GaussianGamma>,
        CanGetLogAverageOfPower<GaussianGamma>,
        CanGetAverageLog<GaussianGamma>,
        Sampleable<Tuple<double, double>>,
        CanGetLogNormalizer
    {
        private DenseVector naturalParameters = DenseVector.Zero(4);

        public GaussianGamma(double firstShape, double firstRate, double secondLocation, double secondPrecisionScale)
        {
            this.SetTo(firstShape, firstRate, secondLocation, secondPrecisionScale);
        }

        public GaussianGamma(Vector naturalParameters)
        {
            Debug.Assert(naturalParameters.Count == 4);
            this.naturalParameters.SetTo(naturalParameters);
        }

        public GaussianGamma()
        {
        }

        public static GaussianGamma FromMeanAndVariance(Tuple<double, double> mean, Tuple<double, double> variance)
        {
            GaussianGamma result = new GaussianGamma();
            result.SetMeanAndVariance(mean, variance);
            return result;
        }

        public bool IsPointMass
        {
            get { return this.Point != null ; }
        }

        public Tuple<double, double> Point { get; set; }

        public Vector GetNaturalParameters()
        {
            return this.naturalParameters.Clone();
        }

        public object Clone()
        {
            if (this.IsPointMass) throw new NotImplementedException();

            GaussianGamma result = new GaussianGamma();
            result.naturalParameters.SetTo(this.naturalParameters);
            return result;
        }

        public double MaxDiff(object that)
        {
            throw new NotImplementedException();
        }

        public bool IsProper()
        {
            if (this.IsPointMass) throw new NotImplementedException();
            
            double firstShape, firstRate, secondLocation, secondPrecisionScale;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);
            return firstShape > 1 && firstRate > 0 && secondPrecisionScale > 0;
        }

        public bool IsUniform()
        {
            if (this.IsPointMass) throw new NotImplementedException();

            return this.naturalParameters.EqualsAll(0);
        }

        public void SetToUniform()
        {
            if (this.IsPointMass) throw new NotImplementedException();
            
            this.naturalParameters.SetAllElementsTo(0);
        }

        public Tuple<double, double> GetMean()
        {
            if (this.IsPointMass) throw new NotImplementedException();
            if (!this.IsProper()) throw new ImproperDistributionException(this);
            
            double firstShape, firstRate, secondLocation, secondPrecisionScale;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            return Tuple.Create(firstShape / firstRate, secondLocation);
        }

        public Tuple<double, double> GetMode()
        {
            if (this.IsPointMass) throw new NotImplementedException();
            if (!this.IsProper()) throw new ImproperDistributionException(this);

            double firstShape, firstRate, secondLocation, secondPrecisionScale;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            return Tuple.Create((firstShape - 0.5) / firstRate, secondLocation);
        }

        public Tuple<double, double> GetVariance()
        {
            if (this.IsPointMass) throw new NotImplementedException();
            if (!this.IsProper()) throw new ImproperDistributionException(this);
            
            double firstShape, firstRate, secondLocation, secondPrecisionScale;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);
            return Tuple.Create(firstShape / (firstRate * firstRate), firstRate / ((firstShape - 1) * secondPrecisionScale));
        }

        public void SetMeanAndVariance(Tuple<double, double> mean, Tuple<double, double> variance)
        {
            if (this.IsPointMass) throw new NotImplementedException();

            double secondMean = mean.Item2;
            double firstRate = mean.Item1 / variance.Item1;
            double firstShape = mean.Item1 * firstRate;
            double secondPrecisionScale = firstRate / (variance.Item2 * (firstShape - 1));
            this.SetTo(firstShape, firstRate, secondMean, secondPrecisionScale);
        }

        public double GetLogProb(Tuple<double, double> value)
        {
            if (this.IsPointMass) throw new NotImplementedException();
            if (!this.IsProper()) throw new ImproperDistributionException(this);
            
            Vector naturalStatistics = ExtractNaturalStatistics(value);
            return this.naturalParameters.Inner(naturalStatistics) - this.GetLogNormalizer();
        }

        public void SetTo(double firstShape, double firstRate, double secondLocation, double secondPrecisionScale)
        {
            if (this.IsPointMass) throw new NotImplementedException();
            
            this.naturalParameters[0] = firstShape - 0.5;
            this.naturalParameters[1] = -firstRate - secondPrecisionScale * secondLocation * secondLocation * 0.5;
            this.naturalParameters[2] = secondLocation * secondPrecisionScale;
            this.naturalParameters[3] = -secondPrecisionScale * 0.5;
        }

        public void SetTo(GaussianGamma value)
        {
            if (this.IsPointMass || value.IsPointMass) throw new NotImplementedException();

            this.naturalParameters.SetTo(value.naturalParameters);
        }

        public void SetToSum(double weight1, GaussianGamma value1, double weight2, GaussianGamma value2)
        {
            if (this.IsPointMass || value1.IsPointMass || value2.IsPointMass) throw new NotImplementedException();

            double weightSum = weight1 + weight2;
            weight1 /= weightSum;
            weight2 /= weightSum;

            Vector moments1 = value1.ComputeNaturalStatisticMoments();
            Vector moments2 = value2.ComputeNaturalStatisticMoments();
            Vector momentsToMatch = Vector.Zero(4);
            momentsToMatch.SetToSum(weight1, moments1, weight2, moments2);

            Gamma distributionOverFirst = Gamma.FromMeanAndMeanLog(momentsToMatch[1], momentsToMatch[0]);
            double secondLocation = momentsToMatch[2] / distributionOverFirst.Shape * distributionOverFirst.Rate;
            double secondPrecisionScale = 1.0 / (momentsToMatch[3] - momentsToMatch[2] * secondLocation);

            this.SetTo(distributionOverFirst.Shape, distributionOverFirst.Rate, secondLocation, secondPrecisionScale);

            Debug.Assert(this.ComputeNaturalStatisticMoments().MaxDiff(momentsToMatch) < 1e-6);
        }

        public void SetToProduct(GaussianGamma a, GaussianGamma b)
        {
            if (this.IsPointMass || a.IsPointMass || b.IsPointMass) throw new NotImplementedException();
            
            this.naturalParameters.SetToSum(a.naturalParameters, b.naturalParameters);
        }

        public void SetToRatio(GaussianGamma numerator, GaussianGamma denominator, bool forceProper = false)
        {
            if (this.IsPointMass || numerator.IsPointMass || denominator.IsPointMass || forceProper) throw new NotImplementedException();

            this.naturalParameters.SetToDifference(numerator.naturalParameters, denominator.naturalParameters);
        }
        
        public void SetToPower(GaussianGamma value, double exponent)
        {
            if (this.IsPointMass || value.IsPointMass) throw new NotImplementedException();

            throw new NotImplementedException();
        }

        public double GetLogAverageOf(GaussianGamma that)
        {
            throw new NotImplementedException();
        }

        public double GetLogAverageOfPower(GaussianGamma that, double power)
        {
            throw new NotImplementedException();
        }

        public double GetAverageLog(GaussianGamma that)
        {
            throw new NotImplementedException();
        }

        public double GetLogNormalizer()
        {
            if (this.IsPointMass) throw new NotImplementedException();
            if (!this.IsProper()) throw new ImproperDistributionException(this);
            
            double firstShape, firstRate, secondLocation, secondPrecisionScale;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);
            double result =
                - firstShape * Math.Log(firstRate)
                - 0.5 * Math.Log(secondPrecisionScale)
                + MMath.LnSqrt2PI
                + MMath.GammaLn(firstShape);
            Debug.Assert(!double.IsNaN(result));

            return result;
        }

        public Tuple<double, double> Sample(Tuple<double, double> result)
        {
            throw new NotImplementedException();
        }

        public Tuple<double, double> Sample()
        {
            double firstShape, firstRate, secondLocation, secondPrecisionScale;
            ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            Gamma firstDistr = Gamma.FromShapeAndRate(firstShape, firstRate);
            double first = firstDistr.Sample();
            Gaussian secondDistr = Gaussian.FromMeanAndPrecision(secondLocation, first * secondPrecisionScale);
            double second = secondDistr.Sample();

            return Tuple.Create(first, second);
        }

        private void ExtractParameters(
            out double firstShape, out double firstRate, out double secondLocation, out double secondPrecisionScale)
        {
            if (this.IsPointMass) throw new NotImplementedException();
            
            firstShape = this.naturalParameters[0] + 0.5;
            secondPrecisionScale = -2 * this.naturalParameters[3];
            secondLocation = this.naturalParameters[2] / secondPrecisionScale;
            firstRate = -this.naturalParameters[1] - 0.5 * this.naturalParameters[2] * secondLocation;
        }

        private Vector ComputeNaturalStatisticMoments()
        {
            if (this.IsPointMass) throw new NotImplementedException();
            
            double firstShape, firstRate, secondLocation, secondPrecisionScale;
            this.ExtractParameters(out firstShape, out firstRate, out secondLocation, out secondPrecisionScale);

            Vector result = Vector.Zero(4);
            result[0] = MMath.Digamma(firstShape) - Math.Log(firstRate);
            result[1] = firstShape / firstRate;
            result[2] = secondLocation * result[1];
            result[3] = 1.0 / secondPrecisionScale + secondLocation * result[2];

            return result;
        }

        private static Vector ExtractNaturalStatistics(Tuple<double, double> value)
        {
            double value1 = value.Item1;
            double value2 = value.Item2;
            double value1Times2 = value1 * value2;
            return Vector.FromArray(Math.Log(value1), value1, value1Times2, value1Times2 * value2);
        }
    }
}
