using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Utils;
using System;
using System.Collections.Generic;

namespace SegmentationGrid
{
    using GaussianArray1D = DistributionStructArray<Gaussian, double>;
    using GaussianArray2D = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using GammaArray1D = DistributionStructArray<Gamma, double>;
    using GammaArray2D = DistributionRefArray<DistributionStructArray<Gamma, double>, double[]>;
    using WishartArray1D = DistributionRefArray<Wishart, PositiveDefiniteMatrix>;

    internal class ImageModelFactorized
    {   
        public Range ObservationRange { get; private set; }

        public Range XYRange;

        public Range WidthRange { get; private set; }

        public Range HeightRange { get; private set; }

        public Range ShapePartRange { get; private set; }

        // Shape model

        public Variable<GaussianArray1D> ShapeLocationMeanPrior { get; private set; }

        public Variable<GammaArray1D> ShapeLocationPrecisionPrior { get; private set; }

        public VariableArray<double> ShapeLocationMean { get; private set; }

        public VariableArray<double> ShapeLocationPrecision { get; private set; }
        
        public VariableArray<double> ShapeLocation { get; private set; }

        public VariableArray<VariableArray<double>, double[][]> ShapePartLocation { get; private set; }

        public Variable<WishartArray1D> ShapePartOrientationRatePriors { get; private set; }

        public VariableArray<PositiveDefiniteMatrix> ShapePartOrientationPriorRates { get; private set; }

        public Variable<double> ShapePartOrientationShape { get; private set; }
        
        public VariableArray<PositiveDefiniteMatrix> ShapePartOrientation { get; private set; }

        public Variable<GaussianArray2D> ShapePartOffsetMeanPriors { get; private set; }

        public Variable<GammaArray2D> ShapePartOffsetPrecisionPriors { get; private set; }

        public VariableArray<VariableArray<double>, double[][]> ShapePartOffsetMeans { get; private set; }

        public VariableArray<VariableArray<double>, double[][]> ShapePartOffsetPrecisions { get; private set; }

        // Observation noise model

        public Variable<Beta> ObservationNoiseProbabilityPrior { get; private set; }

        public Variable<double> ObservationNoiseProbability { get; private set; }

        // Grid specification

        public VariableArray2D<Vector> PixelCoords { get; private set; }

        public VariableArray<VariableArray2D<bool>, bool[][,]> Labels { get; private set; }

        public VariableArray<VariableArray2D<bool>, bool[][,]> ObservedLabels { get; private set; }

        public Variable<double> PottsPenalty { get; private set; }

        public InferenceEngine Engine { get; private set; }

        public class ShapeModelBelief
        {
            public ShapeModelBelief(
                GaussianArray1D shapeLocationMean,
                GammaArray1D shapeLocationPrecision,
                GaussianArray2D shapeOffsetMeanPosteriors,
                GammaArray2D shapeOffsetPrecPosteriors,
                WishartArray1D shapeOrientationRatePosteriors)
            {
                this.ShapeLocationMean = shapeLocationMean;
                this.ShapeLocationPrecision = shapeLocationPrecision;
                this.ShapePartOffsetMeans = shapeOffsetMeanPosteriors;
                this.ShapePartOffsetPrecisions = shapeOffsetPrecPosteriors;
                this.ShapePartOrientationRates = shapeOrientationRatePosteriors;
            }

            public GaussianArray1D ShapeLocationMean { get; private set; }

            public GammaArray1D ShapeLocationPrecision { get; private set; }
            
            public GaussianArray2D ShapePartOffsetMeans { get; private set; }

            public GammaArray2D ShapePartOffsetPrecisions { get; private set; }

            public WishartArray1D ShapePartOrientationRates { get; private set; }
        }

        private GaussianArray2D CreateGaussianArray2D(Gaussian dist, int length1, int length2)
        {
            GaussianArray2D result = new GaussianArray2D(length1);
            for (int i = 0; i < length1; ++i)
            {
                result[i] = new GaussianArray1D(Util.ArrayInit(length2, j => dist));
            }

            return result;
        }

        private GammaArray2D CreateGammaArray2D(Gamma dist, int length1, int length2)
        {
            GammaArray2D result = new GammaArray2D(length1);
            for (int i = 0; i < length1; ++i)
            {
                result[i] = new GammaArray1D(Util.ArrayInit(length2, j => dist));
            }

            return result;
        }

        public ImageModelFactorized(int width, int height, int shapePartCount, int observationCount)
        {
            this.Engine = new InferenceEngine();
            this.Engine.NumberOfIterations = 300;
            this.Engine.Compiler.RecommendedQuality = QualityBand.Unknown;
            this.Engine.Compiler.RequiredQuality = QualityBand.Unknown;
            this.Engine.Compiler.GenerateInMemory = false;
            this.Engine.Compiler.WriteSourceFiles = true;
            this.Engine.Compiler.IncludeDebugInformation = true;
            this.Engine.Compiler.UseSerialSchedules = true;
            
            this.ObservationRange = new Range(observationCount).Named("observation_range");
            this.XYRange = new Range(2).Named("xy_range");
            this.WidthRange = new Range(width).Named("width_range");
            this.HeightRange = new Range(height).Named("height_range");
            this.ShapePartRange = new Range(shapePartCount).Named("shape_part_range");

            //this.ObservationRange.AddAttribute(new Sequential());
            //this.WidthRange.AddAttribute(new Sequential());
            //this.HeightRange.AddAttribute(new Sequential());

            this.ShapeLocationMeanPrior = Variable.New<GaussianArray1D>().Named("shape_location_mean_prior");
            this.ShapeLocationMeanPrior.ObservedValue = new GaussianArray1D(Util.ArrayInit(2, j => Gaussian.FromMeanAndVariance(0.5, 0.3 * 0.3)));
            this.ShapeLocationMean = Variable.Array<double>(this.XYRange).Named("shape_location_mean");
            this.ShapeLocationMean.SetTo(Variable<double[]>.Random(this.ShapeLocationMeanPrior));

            this.ShapeLocationPrecisionPrior = Variable.New<GammaArray1D>().Named("shape_location_prec_prior");
            this.ShapeLocationPrecisionPrior.ObservedValue = new GammaArray1D(Util.ArrayInit(2, j => Gamma.FromMeanAndVariance(1.0 / (0.3 * 0.3), 10 * 10)));
            this.ShapeLocationPrecision = Variable.Array<double>(this.XYRange).Named("shape_location_prec");
            this.ShapeLocationPrecision.SetTo(Variable<double[]>.Random(this.ShapeLocationPrecisionPrior));
            
            this.ShapePartOffsetMeanPriors = Variable.New<GaussianArray2D>().Named("shape_offset_mean_prior");
            this.ShapePartOffsetMeanPriors.ObservedValue = CreateGaussianArray2D(Gaussian.FromMeanAndVariance(0.0, 0.3 * 0.3), shapePartCount, 2);

            this.ShapePartOffsetPrecisionPriors = Variable.New<GammaArray2D>().Named("shape_offset_prec_prior");
            this.ShapePartOffsetPrecisionPriors.ObservedValue = CreateGammaArray2D(Gamma.FromMeanAndVariance(1.0 / (0.01 * 0.01), 50 * 50), shapePartCount, 2);

            this.ShapePartOffsetMeans =
                Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange).Named("shape_part_offset_mean");
            this.ShapePartOffsetMeans.SetTo(Variable<double[][]>.Random(this.ShapePartOffsetMeanPriors));

            this.ShapePartOffsetPrecisions =
                Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange).Named("shape_part_offset_prec");
            this.ShapePartOffsetPrecisions.SetTo(Variable<double[][]>.Random(this.ShapePartOffsetPrecisionPriors));

            const double DefaultShapePartOrientationShape = 500;
            const double DefaultOrientationPriorShape = 5.0;
            const double MeanEllipseHalfSize = 0.2;
            
            this.ShapePartOrientationRatePriors = Variable.New<WishartArray1D>().Named("shape_part_orientation_prior");
            this.ShapePartOrientationRatePriors.ObservedValue = new WishartArray1D(Util.ArrayInit(
                shapePartCount,
                i => Wishart.FromShapeAndRate(
                    DefaultOrientationPriorShape, PositiveDefiniteMatrix.IdentityScaledBy(2, DefaultOrientationPriorShape / (DefaultShapePartOrientationShape * MeanEllipseHalfSize * MeanEllipseHalfSize)))));

            this.ShapePartOrientationPriorRates = Variable.Array<PositiveDefiniteMatrix>(this.ShapePartRange).Named("shape_part_prior_rate");
            this.ShapePartOrientationPriorRates.SetTo(Variable<PositiveDefiniteMatrix[]>.Random(this.ShapePartOrientationRatePriors));

            this.ShapePartOrientationShape = Variable.Observed(DefaultShapePartOrientationShape);
            
            this.ObservationNoiseProbabilityPrior = Variable.New<Beta>().Named("observation_noise_prob_prior");
            this.ObservationNoiseProbabilityPrior.ObservedValue = new Beta(1, 50);
            this.ObservationNoiseProbability = Variable.Random<double, Beta>(this.ObservationNoiseProbabilityPrior).Named("observation_noise_prob");

            this.PottsPenalty = Variable.New<double>();
            this.PottsPenalty.ObservedValue = 0.9;

            this.PixelCoords = Variable.Array<Vector>(this.WidthRange, this.HeightRange).Named("pixel_coords");
            this.PixelCoords.ObservedValue = Util.ArrayInit(width, height, (i, j) => Vector.FromArray((i + 0.5) / width, (j + 0.5) / height));

            this.ObservedLabels =
                Variable.Array<VariableArray2D<bool>, bool[][,]>(Variable.Array<bool>(this.WidthRange, this.HeightRange), this.ObservationRange)
                        .Named("observed_labels");

            using (Variable.ForEach(this.ObservationRange))
            {
                using (Variable.Repeat(1000.0))
                {
                    this.ShapeLocation = Variable.Array<double>(this.XYRange).Named("shape_location");
                    this.ShapeLocation[this.XYRange] = Variable.GaussianFromMeanAndPrecision(this.ShapeLocationMean[this.XYRange], this.ShapeLocationPrecision[this.XYRange]);

                    this.ShapePartLocation = Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange).Named("shape_part_location");
                    this.ShapePartLocation.AddAttribute(new PointEstimate());
                    this.ShapePartLocation.InitialiseTo(Distribution<double>.Array(Util.ArrayInit(shapePartCount, i => Util.ArrayInit(2, j => Gaussian.PointMass(0.45 + 0.1 * Rand.Double())))));
                    this.ShapePartLocation[this.ShapePartRange][this.XYRange] =
                        this.ShapeLocation[this.XYRange] +
                        Variable.GaussianFromMeanAndPrecision(this.ShapePartOffsetMeans[this.ShapePartRange][this.XYRange], this.ShapePartOffsetPrecisions[this.ShapePartRange][this.XYRange])
                        .Named("shape_part_offset");

                    this.ShapePartOrientation = Variable.Array<PositiveDefiniteMatrix>(this.ShapePartRange).Named("shape_part_orientation");
                    this.ShapePartOrientation.AddAttribute(new PointEstimate());
                    this.ShapePartOrientation.InitialiseTo(Distribution<PositiveDefiniteMatrix>.Array(
                        Util.ArrayInit(shapePartCount, i => Wishart.PointMass(PositiveDefiniteMatrix.IdentityScaledBy(2, (0.9 + Rand.Double() * 0.2) / (MeanEllipseHalfSize * MeanEllipseHalfSize))))));
                    this.ShapePartOrientation[this.ShapePartRange] = Variable.WishartFromShapeAndRate(
                        this.ShapePartOrientationShape, this.ShapePartOrientationPriorRates[this.ShapePartRange]);
                    
                    var labels = Variable.Array<bool>(this.WidthRange, this.HeightRange).Named("labels");

                    using (Variable.ForEach(this.WidthRange))
                    using (Variable.ForEach(this.HeightRange))
                    {
                        var labelsByPart = Variable.Array<bool>(this.ShapePartRange).Named("labels_by_part");

                        using (Variable.ForEach(this.ShapePartRange))
                        {
                            labelsByPart[this.ShapePartRange] = Variable<bool>.Factor(
                                ShapeFactors.LabelFromShape,
                                this.PixelCoords[this.WidthRange, this.HeightRange],
                                this.ShapePartLocation[this.ShapePartRange][0],
                                this.ShapePartLocation[this.ShapePartRange][1],
                                this.ShapePartOrientation[this.ShapePartRange]);
                        }

                        labels[this.WidthRange, this.HeightRange] = Variable<bool>.Factor(Factors.AnyTrue, labelsByPart);

                        using (Variable.If(labels[this.WidthRange, this.HeightRange]))
                        {
                            this.ObservedLabels[this.ObservationRange][this.WidthRange, this.HeightRange] =
                                !Variable.Bernoulli(this.ObservationNoiseProbability);
                        }

                        using (Variable.IfNot(labels[this.WidthRange, this.HeightRange]))
                        {
                            this.ObservedLabels[this.ObservationRange][this.WidthRange, this.HeightRange] =
                                Variable.Bernoulli(this.ObservationNoiseProbability);
                        }

                        // TODO: uncomment for Potts
                        //using (Variable.If(widthIterationBlock.Index > 0))
                        //{
                        //    Variable.Constrain(
                        //        GridFactors.Potts,
                        //        this.Labels[widthIterationBlock.Index - 1, heightIterationBlock.Index],
                        //        this.Labels[widthIterationBlock.Index, heightIterationBlock.Index],
                        //        this.PottsPenalty);
                        //}

                        //using (Variable.If(heightIterationBlock.Index > 0))
                        //{
                        //    Variable.Constrain(
                        //        GridFactors.Potts,
                        //        this.Labels[widthIterationBlock.Index, heightIterationBlock.Index - 1],
                        //        this.Labels[widthIterationBlock.Index, heightIterationBlock.Index],
                        //        this.PottsPenalty);
                        //}
                    }
                }
            }
        }

        public ShapeModelBelief InferShapeModel()
        {
            return new ShapeModelBelief(
                this.Engine.Infer<GaussianArray1D>(this.ShapeLocationMean),
                this.Engine.Infer<GammaArray1D>(this.ShapeLocationPrecision),
                this.Engine.Infer<GaussianArray2D>(this.ShapePartOffsetMeans),
                this.Engine.Infer<GammaArray2D>(this.ShapePartOffsetPrecisions),
                this.Engine.Infer<WishartArray1D>(this.ShapePartOrientationPriorRates));
        }

        public void SetShapeModel(ShapeModelBelief shapeModelBelief)
        {
            this.ShapeLocationMeanPrior.ObservedValue = shapeModelBelief.ShapeLocationMean;
            this.ShapeLocationPrecisionPrior.ObservedValue = shapeModelBelief.ShapeLocationPrecision;
            this.ShapePartOffsetMeanPriors.ObservedValue = shapeModelBelief.ShapePartOffsetMeans;
            this.ShapePartOffsetPrecisionPriors.ObservedValue = shapeModelBelief.ShapePartOffsetPrecisions;
            this.ShapePartOrientationRatePriors.ObservedValue = shapeModelBelief.ShapePartOrientationRates;
        }

        public bool[][,] Sample(int sampleCount, bool withNoise = true)
        {
            var labels = new bool[sampleCount][,];
            var noisyLabels = new bool[sampleCount][,];
            var locations = new double[sampleCount][][];
            var orientations = new PositiveDefiniteMatrix[sampleCount][];

            double[] shapeLocationMean = this.ShapeLocationMeanPrior.ObservedValue.Sample();
            double[] shapeLocationPrecision = this.ShapeLocationPrecisionPrior.ObservedValue.Sample();
            double[][] shapeOffsetMeans = this.ShapePartOffsetMeanPriors.ObservedValue.Sample();
            double[][] shapeOffsetPrecisions = this.ShapePartOffsetPrecisionPriors.ObservedValue.Sample();
            PositiveDefiniteMatrix[] shapeOrientationRates = this.ShapePartOrientationRatePriors.ObservedValue.Sample();
            
            for (int sample = 0; sample < sampleCount; ++sample)
            {
                // Sample locations from priors
                double locationX = Gaussian.Sample(shapeLocationMean[0], shapeLocationPrecision[0]);
                double locationY = Gaussian.Sample(shapeLocationMean[1], shapeLocationPrecision[1]);

                // Sample shape part locations
                locations[sample] = new double[this.ShapePartRange.SizeAsInt][];
                orientations[sample] = new PositiveDefiniteMatrix[this.ShapePartRange.SizeAsInt];

                for (int i = 0; i < this.ShapePartRange.SizeAsInt; ++i)
                {
                    double x = locationX + Gaussian.FromMeanAndPrecision(shapeOffsetMeans[i][0], shapeOffsetPrecisions[i][1]).Sample();
                    double y = locationY + Gaussian.FromMeanAndPrecision(shapeOffsetMeans[i][1], shapeOffsetPrecisions[i][1]).Sample();
                    PositiveDefiniteMatrix orientation = Wishart.SampleFromShapeAndRate(this.ShapePartOrientationShape.ObservedValue, shapeOrientationRates[i]);

                    locations[sample][i] = new[] { x, y };
                    orientations[sample][i] = orientation;
                }

                labels[sample] = new bool[this.WidthRange.SizeAsInt, this.HeightRange.SizeAsInt];
                noisyLabels[sample] = new bool[this.WidthRange.SizeAsInt, this.HeightRange.SizeAsInt];
                for (int i = 0; i < this.WidthRange.SizeAsInt; ++i)
                {
                    for (int j = 0; j < this.HeightRange.SizeAsInt; ++j)
                    {
                        labels[sample][i, j] = false;
                        for (int k = 0; k < this.ShapePartRange.SizeAsInt; ++k)
                        {
                            labels[sample][i, j] |= ShapeFactors.LabelFromShape(
                                this.PixelCoords.ObservedValue[i, j],
                                locations[sample][k][0],
                                locations[sample][k][1],
                                orientations[sample][k]);
                        }

                        noisyLabels[sample][i, j] = Rand.Double() >= this.ObservationNoiseProbability.ObservedValue ? labels[sample][i, j] : !labels[sample][i, j];
                    }
                }
            }

            return withNoise ? noisyLabels : labels;
        }

        public IVariable[] GetVariablesForSegmentation()
        {
            return new IVariable[] { this.Labels };
        }

        public IVariable[] GetVariablesForShapeDetection()
        {
            return new IVariable[] { this.ShapePartLocation };
        }

        public IVariable[] GetVariablesForShapeModelLearning()
        {
            return new IVariable[]
            {
                this.ShapePartLocation,
                this.ShapePartOrientation,
                this.ShapePartOffsetMeans,
                this.ShapePartOffsetPrecisions,
                this.ShapePartOrientationPriorRates
            };
        }

        private static PositiveDefiniteMatrix Diagonal(Vector diag)
        {
            PositiveDefiniteMatrix result = PositiveDefiniteMatrix.Identity(diag.Count);
            for (int i = 0; i < diag.Count; ++i)
            {
                result[i, i] = diag[i];
            }

            return result;
        }
    }
}