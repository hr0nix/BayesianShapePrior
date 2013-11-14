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
    internal class ImageModelFactorized
    {   
        public Range ObservationRange { get; private set; }

        public Range XYRange;

        public Range WidthRange { get; private set; }

        public Range HeightRange { get; private set; }

        public Range ShapePartRange { get; private set; }

        // Shape model

        public VariableArray<Gaussian> ShapeLocationPrior { get; private set; }

        public VariableArray<VariableArray<double>, double[][]> ShapeLocation { get; private set; }

        public VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> ShapePartLocation { get; private set; }

        public Variable<Wishart> ShapePartOrientationRatePrior { get; private set; }

        public VariableArray<PositiveDefiniteMatrix> ShapePartOrientationPriorRates { get; private set; }

        public Variable<double> ShapePartOrientationShape { get; private set; }
        
        public VariableArray<VariableArray<PositiveDefiniteMatrix>, PositiveDefiniteMatrix[][]> ShapePartOrientation { get; private set; }

        public VariableArray<VariableArray<Gaussian>, Gaussian[][]> ShapePartOffsetMeanPriors { get; private set; }

        public VariableArray<VariableArray<Gamma>, Gamma[][]> ShapePartOffsetPrecisionPriors { get; private set; }

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

        public ImageModelFactorized(int width, int height, int shapePartCount, int observationCount)
        {
            this.ObservationRange = new Range(observationCount).Named("observation_range");
            this.XYRange = new Range(2).Named("xy_range");
            this.WidthRange = new Range(width).Named("width_range");
            this.HeightRange = new Range(height).Named("height_range");
            this.ShapePartRange = new Range(shapePartCount).Named("shape_part_range");

            //this.ObservationRange.AddAttribute(new Sequential());
            //this.WidthRange.AddAttribute(new Sequential());
            //this.HeightRange.AddAttribute(new Sequential());

            this.ShapePartOffsetMeanPriors =
                Variable.Array(Variable.Array<Gaussian>(this.XYRange), this.ShapePartRange).Named("shape_offset_mean_prior");
            this.ShapePartOffsetMeanPriors.ObservedValue = Util.ArrayInit(
                shapePartCount, i => Util.ArrayInit(2, j => Gaussian.FromMeanAndVariance(0.0, 0.3 * 0.3)));

            this.ShapePartOffsetPrecisionPriors =
                Variable.Array(Variable.Array<Gamma>(this.XYRange), this.ShapePartRange).Named("shape_offset_prec_prior");
            this.ShapePartOffsetPrecisionPriors.ObservedValue = Util.ArrayInit(
                shapePartCount, i => Util.ArrayInit(2, j => Gamma.FromMeanAndVariance(1.0 / (0.01 * 0.01), 50 * 50)));

            this.ShapePartOffsetMeans =
                Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange).Named("shape_part_offset_mean");
            this.ShapePartOffsetMeans[this.ShapePartRange][this.XYRange] =
                Variable<double>.Random(this.ShapePartOffsetMeanPriors[this.ShapePartRange][this.XYRange]);

            // Break symmetry by ordering offset means
            //using (var iterationBlock = Variable.ForEach(this.ShapePartRange))
            //{
            //    using (Variable.If(iterationBlock.Index > 0))
            //    {
            //        Variable<int> index = iterationBlock.Index;
            //        //Variable.ConstrainTrue(
            //        //    (this.ShapePartOffsetMeans[index][0] > this.ShapePartOffsetMeans[index - 1][0]) |
            //        //    (this.ShapePartOffsetMeans[index][0] <= this.ShapePartOffsetMeans[index - 1][0] & this.ShapePartOffsetMeans[index][1] > this.ShapePartOffsetMeans[index][1]));
            //        Variable.ConstrainPositive(this.ShapePartOffsetMeans[index][1] - this.ShapePartOffsetMeans[index - 1][1]);
            //    }
            //}
            // TODO: workaround, the code above does not compile
            //Variable.ConstrainTrue(this.ShapePartOffsetMeans[0][1] > this.ShapePartOffsetMeans[1][1]);

            this.ShapePartOffsetPrecisions =
                Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange).Named("shape_part_offset_prec");
            this.ShapePartOffsetPrecisions[this.ShapePartRange][this.XYRange] =
                Variable<double>.Random(this.ShapePartOffsetPrecisionPriors[this.ShapePartRange][this.XYRange]);

            this.ShapeLocationPrior = Variable.Array<Gaussian>(this.XYRange).Named("shape_location_prior");
            this.ShapeLocationPrior.ObservedValue = Util.ArrayInit(2, j => Gaussian.FromMeanAndVariance(0.5, 0.3 * 0.3));

            this.ShapeLocation = Variable.Array(Variable.Array<double>(this.XYRange), this.ObservationRange).Named("shape_location");
            this.ShapeLocation[this.ObservationRange][this.XYRange] =
                Variable<double>.Random(this.ShapeLocationPrior[this.XYRange]).ForEach(this.ObservationRange);

            this.ShapePartLocation = Variable
                .Array(Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange), this.ObservationRange).Named("shape_part_location");
            this.ShapePartLocation.AddAttribute(new PointEstimate());
            this.ShapePartLocation.InitialiseTo(Distribution<double>.Array(Util.ArrayInit(observationCount, i => Util.ArrayInit(shapePartCount, j => Util.ArrayInit(2, k => Gaussian.PointMass(0.45 + 0.1 * Rand.Double()))))));

            const double DefaultShapePartOrientationShape = 500;
            const double DefaultOrientationPriorShape = 5.0;
            const double MeanEllipseHalfSize = 0.2;
            
            this.ShapePartOrientationRatePrior = Variable.New<Wishart>().Named("shape_part_orientation_prior");
            this.ShapePartOrientationRatePrior.ObservedValue = Wishart.FromShapeAndRate(
                    DefaultOrientationPriorShape, PositiveDefiniteMatrix.IdentityScaledBy(2, DefaultOrientationPriorShape / (DefaultShapePartOrientationShape * MeanEllipseHalfSize * MeanEllipseHalfSize)));

            this.ShapePartOrientationPriorRates = Variable.Array<PositiveDefiniteMatrix>(this.ShapePartRange).Named("shape_part_prior_rate");
            this.ShapePartOrientationPriorRates[this.ShapePartRange] = Variable<PositiveDefiniteMatrix>.Random(this.ShapePartOrientationRatePrior).ForEach(this.ShapePartRange);

            this.ShapePartOrientationShape = Variable.Observed(DefaultShapePartOrientationShape);
            this.ShapePartOrientation = Variable.Array(Variable.Array<PositiveDefiniteMatrix>(this.ShapePartRange), this.ObservationRange).Named("shape_part_orientation");
            this.ShapePartOrientation.AddAttribute(new PointEstimate());
            this.ShapePartOrientation.InitialiseTo(Distribution<PositiveDefiniteMatrix>.Array(
                Util.ArrayInit(observationCount, i => Util.ArrayInit(shapePartCount, j => Wishart.PointMass(PositiveDefiniteMatrix.IdentityScaledBy(2, (0.9 + Rand.Double() * 0.2) / (MeanEllipseHalfSize * MeanEllipseHalfSize)))))));

            using (Variable.ForEach(this.ObservationRange))
            {
                this.ShapePartLocation[this.ObservationRange][this.ShapePartRange][this.XYRange] =
                    this.ShapeLocation[this.ObservationRange][this.XYRange] +
                    Variable.GaussianFromMeanAndPrecision(this.ShapePartOffsetMeans[this.ShapePartRange][this.XYRange], this.ShapePartOffsetPrecisions[this.ShapePartRange][this.XYRange])
                    .Named("shape_part_offset");
                this.ShapePartOrientation[this.ObservationRange][this.ShapePartRange] = Variable.WishartFromShapeAndRate(
                    this.ShapePartOrientationShape, this.ShapePartOrientationPriorRates[this.ShapePartRange]);
            }
            
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
            this.ObservedLabels =
                Variable.Array<VariableArray2D<bool>, bool[][,]>(Variable.Array<bool>(this.WidthRange, this.HeightRange), this.ObservationRange)
                        .Named("observed_labels");

            using (Variable.ForEach(this.ObservationRange))
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
                    this.ObservedLabels[this.ObservationRange][this.WidthRange, this.HeightRange] =
                        !Variable.Bernoulli(this.ObservationNoiseProbability);
                }

                using (Variable.IfNot(this.Labels[this.ObservationRange][this.WidthRange, this.HeightRange]))
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

        public void Sample(int sampleCount, out bool[][,] labels, out double[][][] locations, out PositiveDefiniteMatrix[][] orientations)
        {
            labels = new bool[sampleCount][,];
            locations = new double[sampleCount][][];
            orientations = new PositiveDefiniteMatrix[sampleCount][];

            for (int sample = 0; sample < sampleCount; ++sample)
            {
                // Sample locations from priors
                double locationX = this.ShapeLocationPrior.ObservedValue[0].Sample();
                double locationY = this.ShapeLocationPrior.ObservedValue[1].Sample();

                // Sample shape part locations
                locations[sample] = new double[this.ShapePartRange.SizeAsInt][];
                orientations[sample] = new PositiveDefiniteMatrix[this.ShapePartRange.SizeAsInt];

                for (int i = 0; i < this.ShapePartRange.SizeAsInt; ++i)
                {
                    double x = locationX +
                        Gaussian.FromMeanAndPrecision(this.ShapePartOffsetMeans.ObservedValue[i][0], this.ShapePartOffsetPrecisions.ObservedValue[i][1]).Sample();
                    double y = locationY +
                        Gaussian.FromMeanAndPrecision(this.ShapePartOffsetMeans.ObservedValue[i][1], this.ShapePartOffsetPrecisions.ObservedValue[i][1]).Sample();
                    PositiveDefiniteMatrix orientation = Wishart.SampleFromShapeAndRate(
                        this.ShapePartOrientationShape.ObservedValue, this.ShapePartOrientationPriorRates.ObservedValue[i]);

                    locations[sample][i] = new[] { x, y };
                    orientations[sample][i] = orientation;
                }

                labels[sample] = new bool[this.WidthRange.SizeAsInt, this.HeightRange.SizeAsInt];
                for (int i = 0; i < this.WidthRange.SizeAsInt; ++i)
                {
                    for (int j = 0; j < this.HeightRange.SizeAsInt; ++j)
                    {
                        bool label = false;
                        for (int k = 0; k < this.ShapePartRange.SizeAsInt; ++k)
                        {
                            label |= ShapeFactors.LabelFromShape(
                                this.PixelCoords.ObservedValue[i, j],
                                locations[sample][k][0],
                                locations[sample][k][1],
                                orientations[sample][k]);
                        }

                        if (Rand.Double() < this.ObservationNoiseProbability.ObservedValue)
                        {
                            label = !label;
                        }

                        labels[sample][i, j] = label;
                    }
                }
            }
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