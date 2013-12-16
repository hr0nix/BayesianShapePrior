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
    using GaussianArray3D = DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>, double[][]>;
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

        public Range TraitRange { get; private set; }

        // Shape model

        public Variable<GaussianArray3D> ShapePartOffsetWeightPriors { get; private set; }
        
        // range order: shape part, xy, trait
        public VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> ShapePartOffsetWeights { get; private set; }
        
        public Variable<GaussianArray1D> ShapeLocationMeanPrior { get; private set; }

        public Variable<GammaArray1D> ShapeLocationPrecisionPrior { get; private set; }

        public VariableArray<double> ShapeLocationMean { get; private set; }

        public VariableArray<double> ShapeLocationPrecision { get; private set; }
        
        public VariableArray<VariableArray<double>, double[][]> ShapeLocation { get; private set; }

        public VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> ShapePartLocation { get; private set; }

        public Variable<WishartArray1D> ShapePartOrientationRatePriors { get; private set; }

        public VariableArray<PositiveDefiniteMatrix> ShapePartOrientationPriorRates { get; private set; }

        public Variable<double> ShapePartOrientationShape { get; private set; }
        
        public VariableArray<VariableArray<PositiveDefiniteMatrix>, PositiveDefiniteMatrix[][]> ShapePartOrientation { get; private set; }

        public Variable<GammaArray2D> ShapePartOffsetPrecisionPriors { get; private set; }

        public VariableArray<VariableArray<double>, double[][]> ShapePartOffsetPrecisions { get; private set; }

        public Variable<GaussianArray2D> ShapeTraitsPrior { get; private set; }
        
        public VariableArray<VariableArray<double>, double[][]> ShapeTraits { get; private set; }

        // Observation noise model

        public Variable<Beta> ObservationNoiseProbabilityPrior { get; private set; }

        public Variable<double> ObservationNoiseProbability { get; private set; }

        // Grid specification

        public VariableArray2D<Vector> PixelCoords { get; private set; }

        public VariableArray<VariableArray2D<bool>, bool[][,]> Labels { get; private set; }

        public VariableArray<VariableArray2D<bool>, bool[][,]> NoisyLabels { get; private set; }

        public VariableArray<VariableArray2D<Bernoulli>, Bernoulli[][,]> NoisyLabelsConstraint { get; private set; }

        public Variable<double> PottsPenalty { get; private set; }

        public InferenceEngine Engine { get; private set; }

        public class ShapeModelBelief
        {
            public ShapeModelBelief(
                GaussianArray1D shapeLocationMean,
                GammaArray1D shapeLocationPrecision,
                GaussianArray3D shapeOffsetWeights,
                GammaArray2D shapePartOffsetPrecision,
                WishartArray1D shapePartOrientationRates)
            {
                this.ShapeLocationMean = shapeLocationMean;
                this.ShapeLocationPrecision = shapeLocationPrecision;
                this.ShapePartOffsetWeights = shapeOffsetWeights;
                this.ShapePartOffsetPrecisions = shapePartOffsetPrecision;
                this.ShapePartOrientationRates = shapePartOrientationRates;
            }

            public GaussianArray1D ShapeLocationMean { get; private set; }

            public GammaArray1D ShapeLocationPrecision { get; private set; }
            
            public GaussianArray3D ShapePartOffsetWeights { get; private set; }

            public GammaArray2D ShapePartOffsetPrecisions { get; private set; }

            public WishartArray1D ShapePartOrientationRates { get; private set; }
        }

        private GaussianArray2D CreateGaussianArray2D(Gaussian dist, int length1, int length2)
        {
            return new GaussianArray2D(Util.ArrayInit(length1, i => new GaussianArray1D(Util.ArrayInit(length2, j => dist))));
        }

        private GaussianArray3D CreateGaussianArray3D(Gaussian dist, int length1, int length2, int length3)
        {
            return new GaussianArray3D(Util.ArrayInit(length1, i => CreateGaussianArray2D(dist, length2, length3)));
        }

        private GammaArray2D CreateGammaArray2D(Gamma dist, int length1, int length2)
        {
            return new GammaArray2D(Util.ArrayInit(length1, i => new GammaArray1D(Util.ArrayInit(length2, j => dist))));
        }

        public ImageModelFactorized(int width, int height, int shapePartCount, int traitCount, int observationCount)
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
            this.TraitRange = new Range(traitCount).Named("trait_range");

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

            this.ShapePartOffsetWeightPriors = Variable.New<GaussianArray3D>().Named("shape_part_offset_weight_prior");
            this.ShapePartOffsetWeightPriors.ObservedValue = CreateGaussianArray3D(Gaussian.FromMeanAndVariance(0, 1), shapePartCount, 2, traitCount);

            this.ShapePartOffsetWeights = Variable.Array(Variable.Array(Variable.Array<double>(this.TraitRange), this.XYRange), this.ShapePartRange).Named("shape_part_offset_weights");
            this.ShapePartOffsetWeights.SetTo(Variable<double[][][]>.Random(this.ShapePartOffsetWeightPriors));

            this.ShapePartOffsetPrecisionPriors = Variable.New<GammaArray2D>().Named("shape_part_offset_prec_prior");
            this.ShapePartOffsetPrecisionPriors.ObservedValue = CreateGammaArray2D(Gamma.FromMeanAndVariance(1.0 / (0.01 * 0.01), 50 * 50), shapePartCount, 2);

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
           

            this.ShapeLocation = Variable.Array(Variable.Array<double>(this.XYRange), this.ObservationRange).Named("shape_location");
            this.ShapePartLocation = Variable.Array(Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange), this.ObservationRange).Named("shape_part_location");
            this.ShapePartLocation.AddAttribute(new PointEstimate());
            this.ShapePartLocation.InitialiseTo(Distribution<double>.Array(
                Util.ArrayInit(observationCount, i => Util.ArrayInit(shapePartCount, j => Util.ArrayInit(2, k => Gaussian.PointMass(0.45 + 0.1 * Rand.Double()))))));

            this.ShapePartOrientation = Variable.Array(Variable.Array<PositiveDefiniteMatrix>(this.ShapePartRange), this.ObservationRange).Named("shape_part_orientation");
            this.ShapePartOrientation.AddAttribute(new PointEstimate());
            this.ShapePartOrientation.InitialiseTo(Distribution<PositiveDefiniteMatrix>.Array(
                Util.ArrayInit(observationCount, i => Util.ArrayInit(shapePartCount, j => Wishart.PointMass(PositiveDefiniteMatrix.IdentityScaledBy(2, (0.9 + Rand.Double() * 0.2) / (MeanEllipseHalfSize * MeanEllipseHalfSize)))))));

            this.ShapeTraitsPrior = Variable.New<GaussianArray2D>().Named("shape_traits_prior");
            this.ShapeTraitsPrior.ObservedValue = new GaussianArray2D(observationCount);
            int squarePartSize = Math.Min(observationCount, traitCount - 1);
            for (int i = 0; i < observationCount; ++i)
            {
                this.ShapeTraitsPrior.ObservedValue[i] = new GaussianArray1D(traitCount);
                for (int j = 0; j < traitCount; ++j)
                {
                    if (i < squarePartSize && j < squarePartSize)
                    {
                        // Break symmetry in traits
                        this.ShapeTraitsPrior.ObservedValue[i][j] = i == j ? Gaussian.PointMass(1) : Gaussian.PointMass(0);
                    }
                    else if (j == traitCount - 1)
                    {
                        // Weights corresponding to the last trait describe the offset
                        this.ShapeTraitsPrior.ObservedValue[i][j] = Gaussian.PointMass(1);
                    }
                    else
                    {
                        this.ShapeTraitsPrior.ObservedValue[i][j] = Gaussian.FromMeanAndVariance(0, 1);
                    }
                }
            }

            this.ShapeTraits = Variable.Array(Variable.Array<double>(this.TraitRange), this.ObservationRange).Named("shape_traits");
            this.ShapeTraits.SetTo(Variable<double[][]>.Random(this.ShapeTraitsPrior));

            this.ObservationNoiseProbabilityPrior = Variable.New<Beta>().Named("observation_noise_prob_prior");
            this.ObservationNoiseProbabilityPrior.ObservedValue = new Beta(1, 50);
            this.ObservationNoiseProbability = Variable.Random<double, Beta>(this.ObservationNoiseProbabilityPrior).Named("observation_noise_prob");

            this.PottsPenalty = Variable.New<double>();
            this.PottsPenalty.ObservedValue = 0.9;

            this.PixelCoords = Variable.Array<Vector>(this.WidthRange, this.HeightRange).Named("pixel_coords");
            this.PixelCoords.ObservedValue = Util.ArrayInit(width, height, (i, j) => Vector.FromArray((i + 0.5) / width, (j + 0.5) / height));

            this.Labels =
                Variable.Array<VariableArray2D<bool>, bool[][,]>(Variable.Array<bool>(this.WidthRange, this.HeightRange), this.ObservationRange)
                        .Named("labels");
            this.NoisyLabels =
                Variable.Array<VariableArray2D<bool>, bool[][,]>(Variable.Array<bool>(this.WidthRange, this.HeightRange), this.ObservationRange)
                        .Named("noisy_labels");
            this.NoisyLabelsConstraint =
                Variable.Array<VariableArray2D<Bernoulli>, Bernoulli[][,]>(Variable.Array<Bernoulli>(this.WidthRange, this.HeightRange), this.ObservationRange)
                        .Named("noisy_labels_constraint");

            this.NoisyLabelsConstraint.ObservedValue = Util.ArrayInit(observationCount, o => Util.ArrayInit(width, height, (i, j) => Bernoulli.Uniform()));
            
            using (var observationIter = Variable.ForEach(this.ObservationRange))
            {
                //using (Variable.Repeat(1000.0))
                {                    
                    this.ShapeLocation[this.ObservationRange][this.XYRange] = Variable.GaussianFromMeanAndPrecision(this.ShapeLocationMean[this.XYRange], this.ShapeLocationPrecision[this.XYRange]);
                    using (Variable.ForEach(this.ShapePartRange))
                    {
                        var shapePartOffsetMeanTraitWeightProducts = Variable.Array(Variable.Array<double>(this.TraitRange), this.XYRange).Named("shape_part_offset_mean_products");
                        shapePartOffsetMeanTraitWeightProducts[this.XYRange][this.TraitRange] =
                            Variable<double>.Factor(Factor.Product_SHG09, this.ShapeTraits[this.ObservationRange][this.TraitRange], this.ShapePartOffsetWeights[this.ShapePartRange][this.XYRange][this.TraitRange]);

                        var dampedProducts = Variable.Array(Variable.Array<double>(this.TraitRange), this.XYRange).Named("shape_part_offset_mean_products_damped");
                        dampedProducts[this.XYRange][this.TraitRange] = Variable<double>.Factor(Damp.Forward<double>, shapePartOffsetMeanTraitWeightProducts[this.XYRange][this.TraitRange], 0.5);
                        
                        var shapePartOffsetMean = Variable.Array<double>(this.XYRange).Named("shape_part_offset_mean");
                        shapePartOffsetMean[this.XYRange] = Variable.Sum(dampedProducts[this.XYRange]);
                        
                        this.ShapePartLocation[this.ObservationRange][this.ShapePartRange][this.XYRange] =
                            this.ShapeLocation[this.ObservationRange][this.XYRange] +
                            Variable.GaussianFromMeanAndPrecision(shapePartOffsetMean[this.XYRange], this.ShapePartOffsetPrecisions[this.ShapePartRange][this.XYRange]).Named("shape_part_offset");

                        this.ShapePartOrientation[this.ObservationRange][this.ShapePartRange] = Variable.WishartFromShapeAndRate(
                            this.ShapePartOrientationShape, this.ShapePartOrientationPriorRates[this.ShapePartRange]);
                    }

                    using (Variable.ForEach(this.WidthRange))
                    using (Variable.ForEach(this.HeightRange))
                    {
                        var labelsByPart = Variable.Array<bool>(this.ShapePartRange).Named("labels_by_part");

                        using (Variable.ForEach(this.ShapePartRange))
                        {
                            labelsByPart[this.ShapePartRange] = Variable<bool>.Factor(
                                ShapeFactors.LabelFromShape,
                                this.PixelCoords[this.WidthRange, this.HeightRange],
                                this.ShapePartLocation[this.ObservationRange][this.ShapePartRange][0],
                                this.ShapePartLocation[this.ObservationRange][this.ShapePartRange][1],
                                this.ShapePartOrientation[this.ObservationRange][this.ShapePartRange]);
                        }

                        this.Labels[this.ObservationRange][this.WidthRange, this.HeightRange] = Variable<bool>.Factor(Factors.AnyTrue, labelsByPart);

                        using (Variable.If(this.Labels[this.ObservationRange][this.WidthRange, this.HeightRange]))
                        {
                            this.NoisyLabels[this.ObservationRange][this.WidthRange, this.HeightRange] =
                                !Variable.Bernoulli(this.ObservationNoiseProbability);
                        }

                        using (Variable.IfNot(this.Labels[this.ObservationRange][this.WidthRange, this.HeightRange]))
                        {
                            this.NoisyLabels[this.ObservationRange][this.WidthRange, this.HeightRange] =
                                Variable.Bernoulli(this.ObservationNoiseProbability);
                        }

                        Variable.ConstrainEqualRandom(
                            this.NoisyLabels[this.ObservationRange][this.WidthRange, this.HeightRange],
                            this.NoisyLabelsConstraint[this.ObservationRange][this.WidthRange, this.HeightRange]);

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
                this.Engine.Infer<GaussianArray3D>(this.ShapePartOffsetWeights),
                this.Engine.Infer<GammaArray2D>(this.ShapePartOffsetPrecisions),
                this.Engine.Infer<WishartArray1D>(this.ShapePartOrientationPriorRates));
        }

        public void SetShapeModel(ShapeModelBelief shapeModelBelief)
        {
            this.ShapeLocationMeanPrior.ObservedValue = shapeModelBelief.ShapeLocationMean;
            this.ShapeLocationPrecisionPrior.ObservedValue = shapeModelBelief.ShapeLocationPrecision;
            this.ShapePartOffsetWeightPriors.ObservedValue = shapeModelBelief.ShapePartOffsetWeights;
            this.ShapePartOffsetPrecisionPriors.ObservedValue = shapeModelBelief.ShapePartOffsetPrecisions;
            this.ShapePartOrientationRatePriors.ObservedValue = shapeModelBelief.ShapePartOrientationRates;
        }

        public bool[][,] Sample(int sampleCount, bool withNoise, bool withSymmetryBreaking, bool inCenter)
        {
            var labels = new bool[sampleCount][,];
            var noisyLabels = new bool[sampleCount][,];
            var locations = new double[sampleCount][][];
            var orientations = new PositiveDefiniteMatrix[sampleCount][];

            double[] shapeLocationMean = this.ShapeLocationMeanPrior.ObservedValue.Sample();
            double[] shapeLocationPrecision = this.ShapeLocationPrecisionPrior.ObservedValue.Sample();
            double[][] shapePartOffsetPrecisions = this.ShapePartOffsetPrecisionPriors.ObservedValue.Sample();
            double[][][] shapePartOffsetMeanWeights = this.ShapePartOffsetWeightPriors.ObservedValue.Sample();
            PositiveDefiniteMatrix[] shapeOrientationRates = this.ShapePartOrientationRatePriors.ObservedValue.Sample();
            
            for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
            {
                double[] traits = new double[this.TraitRange.SizeAsInt];
                for (int traitIndex = 0; traitIndex < traits.Length; ++traitIndex)
                {
                    if (traitIndex == traits.Length - 1)
                    {
                        traits[traitIndex] = 1;
                    }
                    else if (withSymmetryBreaking && traitIndex < sampleCount)
                    {
                        traits[traitIndex] = traitIndex == sampleIndex ? 1 : 0;
                    }
                    else
                    {
                        traits[traitIndex] = Gaussian.Sample(0, 1);
                    }
                }

                // Sample locations from priors
                double locationX, locationY;
                if (inCenter)
                {
                    locationX = 0.5;
                    locationY = 0.5;
                }
                else
                {
                    locationX = Gaussian.Sample(shapeLocationMean[0], shapeLocationPrecision[0]);
                    locationY = Gaussian.Sample(shapeLocationMean[1], shapeLocationPrecision[1]);
                }

                // Sample shape part locations
                locations[sampleIndex] = new double[this.ShapePartRange.SizeAsInt][];
                orientations[sampleIndex] = new PositiveDefiniteMatrix[this.ShapePartRange.SizeAsInt];

                for (int shapePartIndex = 0; shapePartIndex < this.ShapePartRange.SizeAsInt; ++shapePartIndex)
                {
                    double meanX = Vector.InnerProduct(Vector.FromArray(traits), Vector.FromArray(shapePartOffsetMeanWeights[shapePartIndex][0]));
                    double meanY = Vector.InnerProduct(Vector.FromArray(traits), Vector.FromArray(shapePartOffsetMeanWeights[shapePartIndex][1]));

                    double x = locationX + Gaussian.FromMeanAndPrecision(meanX, shapePartOffsetPrecisions[shapePartIndex][0]).Sample();
                    double y = locationY + Gaussian.FromMeanAndPrecision(meanY, shapePartOffsetPrecisions[shapePartIndex][1]).Sample();
                    PositiveDefiniteMatrix orientation = Wishart.SampleFromShapeAndRate(this.ShapePartOrientationShape.ObservedValue, shapeOrientationRates[shapePartIndex]);

                    locations[sampleIndex][shapePartIndex] = new[] { x, y };
                    orientations[sampleIndex][shapePartIndex] = orientation;
                }

                labels[sampleIndex] = new bool[this.WidthRange.SizeAsInt, this.HeightRange.SizeAsInt];
                noisyLabels[sampleIndex] = new bool[this.WidthRange.SizeAsInt, this.HeightRange.SizeAsInt];
                for (int i = 0; i < this.WidthRange.SizeAsInt; ++i)
                {
                    for (int j = 0; j < this.HeightRange.SizeAsInt; ++j)
                    {
                        labels[sampleIndex][i, j] = false;
                        for (int k = 0; k < this.ShapePartRange.SizeAsInt; ++k)
                        {
                            labels[sampleIndex][i, j] |= ShapeFactors.LabelFromShape(
                                this.PixelCoords.ObservedValue[i, j],
                                locations[sampleIndex][k][0],
                                locations[sampleIndex][k][1],
                                orientations[sampleIndex][k]);
                        }

                        noisyLabels[sampleIndex][i, j] = Rand.Double() >= this.ObservationNoiseProbability.ObservedValue ? labels[sampleIndex][i, j] : !labels[sampleIndex][i, j];
                    }
                }
            }

            return withNoise ? noisyLabels : labels;
        }

        public void OptimizeForShapeModelLearning()
        {
            this.Engine.OptimiseForVariables = new IVariable[]
            {
                this.ShapeLocationMean,
                this.ShapeLocationPrecision,
                this.ShapePartLocation,
                this.ShapePartOrientation,
                this.ShapePartOffsetWeights,
                this.ShapePartOffsetPrecisions,
                this.ShapePartOrientationPriorRates,
            };
        }

        public void OptimizeForLabelCompletion()
        {
            this.Engine.OptimiseForVariables = new IVariable[]
            {
                this.Labels,
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