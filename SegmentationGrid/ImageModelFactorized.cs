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

        public VariableArray<Bernoulli> ShapePartPresentedPrior { get; private set; }

        public VariableArray<bool> ShapePartPresented { get; private set; }

        public VariableArray<Gaussian> ShapeLocationPrior { get; private set; }

        public VariableArray<VariableArray<double>, double[][]> ShapeLocation { get; private set; }

        public VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> ShapePartLocation { get; private set; }

        public VariableArray<PositiveDefiniteMatrix> ShapePartOrientation { get; private set; }

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

            this.ShapePartPresentedPrior = Variable.Array<Bernoulli>(this.ShapePartRange).Named("shape_part_presented_prior");
            this.ShapePartPresentedPrior.ObservedValue = Util.ArrayInit(shapePartCount, i => Bernoulli.Uniform());

            this.ShapePartPresented = Variable.Array<bool>(this.ShapePartRange).Named("shape_part_presented");
            this.ShapePartPresented[this.ShapePartRange] = Variable<bool>.Random(this.ShapePartPresentedPrior[this.ShapePartRange]);

            this.ShapePartOffsetMeanPriors =
                Variable.Array(Variable.Array<Gaussian>(this.XYRange), this.ShapePartRange).Named("shape_offset_mean_prior");
            this.ShapePartOffsetMeanPriors.ObservedValue = Util.ArrayInit(
                shapePartCount, i => Util.ArrayInit(2, j => Gaussian.FromMeanAndVariance(0.0, 0.5 * 0.5)));

            this.ShapePartOffsetPrecisionPriors =
                Variable.Array(Variable.Array<Gamma>(this.XYRange), this.ShapePartRange).Named("shape_offset_prec_prior");
            this.ShapePartOffsetPrecisionPriors.ObservedValue = Util.ArrayInit(
                shapePartCount, i => Util.ArrayInit(2, j => Gamma.FromMeanAndVariance(1.0 / (0.05 * 0.05), 400 * 400)));

            this.ShapePartOffsetMeans =
                Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange).Named("shape_part_offset_mean");
            this.ShapePartOffsetMeans[this.ShapePartRange][this.XYRange] =
                Variable<double>.Random(this.ShapePartOffsetMeanPriors[this.ShapePartRange][this.XYRange]);

            this.ShapePartOffsetPrecisions =
                Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange).Named("shape_part_offset_prec");
            this.ShapePartOffsetPrecisions[this.ShapePartRange][this.XYRange] =
                Variable<double>.Random(this.ShapePartOffsetPrecisionPriors[this.ShapePartRange][this.XYRange]);

            this.ShapeLocationPrior = Variable.Array<Gaussian>(this.XYRange).Named("shape_location_prior");
            this.ShapeLocationPrior.ObservedValue = Util.ArrayInit(2, j => Gaussian.FromMeanAndVariance(0.5, 0.5 * 0.5));

            this.ShapeLocation = Variable.Array(Variable.Array<double>(this.XYRange), this.ObservationRange).Named("shape_location");
            this.ShapeLocation[this.ObservationRange][this.XYRange] =
                Variable<double>.Random(this.ShapeLocationPrior[this.XYRange]).ForEach(this.ObservationRange);

            this.ShapePartLocation = Variable
                .Array(Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange), this.ObservationRange)
                .Named("shape_part_location");
            this.ShapePartLocation[this.ObservationRange][this.ShapePartRange][this.XYRange] =
                this.ShapeLocation[this.ObservationRange][this.XYRange] +
                Variable.GaussianFromMeanAndPrecision(this.ShapePartOffsetMeans[this.ShapePartRange][this.XYRange], this.ShapePartOffsetPrecisions[this.ShapePartRange][this.XYRange])
                        .ForEach(this.ObservationRange)
                        .Named("shape_part_offset");

            // Break symmetry
            var shapePartOffsetMeansTranspose =
                Variable.Array(Variable.Array<double>(this.ShapePartRange), this.XYRange).Named("shape_part_mean_offset_transposed");
            shapePartOffsetMeansTranspose[this.XYRange][this.ShapePartRange] =
                Variable.Copy(this.ShapePartOffsetMeans[this.ShapePartRange][this.XYRange]);
            Variable.ConstrainEqual(Variable.Sum(shapePartOffsetMeansTranspose[this.XYRange]), 0);
            
            //using (Variable.ForEach(this.ObservationRange))
            //{
            //    var shapePartLocationTransposed =
            //        Variable.Array(Variable.Array<double>(this.ShapePartRange), this.XYRange).Named("shape_part_location_transposed");
            //    shapePartLocationTransposed[this.XYRange][this.ShapePartRange] =
            //        Variable.Copy(this.ShapePartLocation[this.ObservationRange][this.ShapePartRange][this.XYRange]);
            //    var shapePartLocationCoordSum =
            //        Variable.Sum(shapePartLocationTransposed[this.XYRange]).Named("shape_part_location_coord_sum");
            //    Variable.ConstrainEqual(shapePartLocationCoordSum, 0);
            //}

            this.ShapePartOrientation = Variable.Array<PositiveDefiniteMatrix>(this.ShapePartRange).Named("shape_part_orientation");

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

            var labelsByPart =
                Variable.Array<VariableArray2D<VariableArray<bool>, bool[,][]>, bool[][,][]>(
                    Variable.Array(Variable.Array<bool>(this.ShapePartRange), this.WidthRange, this.HeightRange), this.ObservationRange)
                        .Named("labels_by_part");
            using (Variable.ForEach(this.ShapePartRange))
            {
                using (Variable.If(this.ShapePartPresented[this.ShapePartRange]))
                {
                    using (Variable.ForEach(this.ObservationRange))
                    using (Variable.ForEach(this.WidthRange))
                    using (Variable.ForEach(this.HeightRange))
                    {
                        const double damping = 0.001;
                        VariableArray<double> dampedShapePartLocations = Variable.Array<double>(this.XYRange);
                        dampedShapePartLocations[this.XYRange] =
                            Variable<double>.Factor(Damp.Backward<double>, this.ShapePartLocation[this.ObservationRange][this.ShapePartRange][this.XYRange], damping)
                                            .Named("damped_shape_part_location");

                        labelsByPart[this.ObservationRange][this.WidthRange, this.HeightRange][this.ShapePartRange] = Variable<bool>.Factor(
                            ShapeFactors.LabelFromShape,
                            this.PixelCoords[this.WidthRange, this.HeightRange],
                            dampedShapePartLocations[0],
                            dampedShapePartLocations[1],
                            this.ShapePartOrientation[this.ShapePartRange]);
                    }
                }

                using (Variable.IfNot(this.ShapePartPresented[this.ShapePartRange]))
                {
                    using (Variable.ForEach(this.WidthRange))
                    using (Variable.ForEach(this.HeightRange))
                    {
                        labelsByPart[this.ObservationRange][this.WidthRange, this.HeightRange][this.ShapePartRange] = false;
                    }
                }
            }

            using (Variable.ForEach(this.ObservationRange))
            using (Variable.ForEach(this.WidthRange))
            using (Variable.ForEach(this.HeightRange))
            {
                this.Labels[this.ObservationRange][this.WidthRange, this.HeightRange] =
                    Variable<bool>.Factor(Factors.AnyTrue, labelsByPart[this.ObservationRange][this.WidthRange, this.HeightRange]);

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

        public bool[][,] Sample(int sampleCount)
        {
            bool[] shapePartPresense = Util.ArrayInit(
                this.ShapePartRange.SizeAsInt, i => this.ShapePartPresentedPrior.ObservedValue[i].Sample());

            bool[][,] result = new bool[sampleCount][,];
            for (int sample = 0; sample < sampleCount; ++sample)
            {
                // Sample locations from priors
                double locationX = this.ShapeLocationPrior.ObservedValue[0].Sample();
                double locationY = this.ShapeLocationPrior.ObservedValue[1].Sample();

                // Sample shape part locations
                List<double> shapePartLocationX = new List<double>();
                List<double> shapePartLocationY = new List<double>();

                for (int i = 0; i < this.ShapePartRange.SizeAsInt; ++i)
                {
                    double x = locationX +
                        Gaussian.FromMeanAndPrecision(this.ShapePartOffsetMeans.ObservedValue[i][0], this.ShapePartOffsetPrecisions.ObservedValue[i][1]).Sample();
                    double y = locationY +
                        Gaussian.FromMeanAndPrecision(this.ShapePartOffsetMeans.ObservedValue[i][1], this.ShapePartOffsetPrecisions.ObservedValue[i][1]).Sample();

                    shapePartLocationX.Add(x);
                    shapePartLocationY.Add(y);
                }

                result[sample] = new bool[this.WidthRange.SizeAsInt, this.HeightRange.SizeAsInt];
                for (int i = 0; i < this.WidthRange.SizeAsInt; ++i)
                {
                    for (int j = 0; j < this.HeightRange.SizeAsInt; ++j)
                    {
                        bool label = false;
                        for (int k = 0; k < this.ShapePartRange.SizeAsInt; ++k)
                        {
                            if (shapePartPresense[k])
                            {
                                label |= ShapeFactors.LabelFromShape(
                                    this.PixelCoords.ObservedValue[i, j],
                                    shapePartLocationX[k],
                                    shapePartLocationY[k],
                                    this.ShapePartOrientation.ObservedValue[k]);
                            }
                        }

                        if (Rand.Double() < this.ObservationNoiseProbability.ObservedValue)
                        {
                            label = !label;
                        }

                        result[sample][i, j] = label;
                    }
                }
            }

            return result;
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
                this.ShapePartLocation /* remove me? */,
                this.ShapePartPresented,
                this.ShapePartOffsetMeans,
                this.ShapePartOffsetPrecisions
            };
        }
    }
}