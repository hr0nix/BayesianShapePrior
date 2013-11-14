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
    internal class ImageModelFullCovariance
    {
        public Range ObservationRange { get; private set; }

        public Range XYRange { get; private set; }

        public Range WidthRange { get; private set; }

        public Range HeightRange { get; private set; }

        public Range ShapePartRange { get; private set; }

        // Shape model

        public VariableArray<Gaussian> ShapeLocationPrior { get; private set; }

        public VariableArray<VariableArray<double>, double[][]> ShapeLocation { get; private set; }

        public VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> ShapePartLocation { get; private set; }

        public VariableArray<Wishart> ShapePartOrientationPrior { get; private set; }
        
        public VariableArray<VariableArray<PositiveDefiniteMatrix>, PositiveDefiniteMatrix[][]> ShapePartOrientation { get; private set; }

        public VariableArray<VariableArray<Gaussian>, Gaussian[][]> ShapePartOffsetMeanPriors { get; private set; }

        public VariableArray<VariableArray<Gamma>, Gamma[][]> ShapePartOffsetPrecisionPriors { get; private set; }

        public VariableArray<VariableArray<double>, double[][]> ShapePartOffsetMeans { get; private set; }

        public VariableArray<VariableArray<double>, double[][]> ShapePartOffsetPrecisions { get; private set; }

        public VariableArray<VariableArray<Tuple<PositiveDefiniteMatrix, Vector>>, Tuple<PositiveDefiniteMatrix, Vector>[][]> ShapePartConfigurationsCombined { get; private set; }

        // Observation noise model

        public Variable<Beta> ObservationNoiseProbabilityPrior { get; private set; }

        public Variable<double> ObservationNoiseProbability { get; private set; }

        // Grid specification

        public VariableArray2D<Vector> PixelCoords { get; private set; }

        public VariableArray<VariableArray2D<bool>, bool[][,]> Labels { get; private set; }

        public VariableArray<VariableArray2D<bool>, bool[][,]> ObservedLabels { get; private set; }

        public Variable<double> PottsPenalty { get; private set; }

        public ImageModelFullCovariance(int width, int height, int shapePartCount, int observationCount)
        {
            this.ObservationRange = new Range(observationCount).Named("observation_range");
            this.XYRange = new Range(2).Named("xy_range");
            this.WidthRange = new Range(width).Named("width_range");
            this.HeightRange = new Range(height).Named("height_range");
            this.ShapePartRange = new Range(shapePartCount).Named("shape_part_range");

            //this.ObservationRange.AddAttribute(new Sequential());
            //this.WidthRange.AddAttribute(new Sequential());
            //this.HeightRange.AddAttribute(new Sequential());
            //this.ShapePartRange.AddAttribute(new Sequential());

            this.ShapePartOffsetMeanPriors =
                Variable.Array(Variable.Array<Gaussian>(this.XYRange), this.ShapePartRange).Named("shape_offset_mean_prior");
            this.ShapePartOffsetMeanPriors.ObservedValue = Util.ArrayInit(
                shapePartCount, i => Util.ArrayInit(2, j => Gaussian.FromMeanAndVariance(0.0, 0.3 * 0.3)));

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
            
            // TODO: remove me | disable learning precisions
            this.ShapePartOffsetPrecisions.ObservedValue =
                Util.ArrayInit(shapePartCount, i => Util.ArrayInit(2, j => 200.0));

            this.ShapeLocationPrior = Variable.Array<Gaussian>(this.XYRange).Named("shape_location_prior");
            this.ShapeLocationPrior.ObservedValue = Util.ArrayInit(2, j => Gaussian.FromMeanAndVariance(0.5, 0.3 * 0.3));

            this.ShapeLocation = Variable.Array(Variable.Array<double>(this.XYRange), this.ObservationRange).Named("shape_location");
            this.ShapeLocation[this.ObservationRange][this.XYRange] =
                Variable<double>.Random(this.ShapeLocationPrior[this.XYRange]).ForEach(this.ObservationRange);

            this.ShapePartLocation = Variable
                .Array(Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange), this.ObservationRange)
                .Named("shape_part_location");
            
            // TODO: no hierarchy of locations, test only
            //var shapePartLocationPrior = Variable.Array(Variable.Array<Gaussian>(this.XYRange), this.ShapePartRange).Named("shape_part_location_prior");
            //shapePartLocationPrior.ObservedValue = Util.ArrayInit(shapePartCount, i => Util.ArrayInit(2, j => Gaussian.FromMeanAndVariance(Rand.Double(), 0.5 * 0.5)));
            //this.ShapePartLocation[this.ObservationRange][this.ShapePartRange][this.XYRange] = Variable<double>.Random(
            //    shapePartLocationPrior[this.ShapePartRange][this.XYRange]).ForEach(this.ObservationRange).Named("shape_part_location");

            // Hierarchy of locations
            this.ShapePartLocation[this.ObservationRange][this.ShapePartRange][this.XYRange] =
                Variable.GaussianFromMeanAndPrecision(
                    this.ShapeLocation[this.ObservationRange][this.XYRange] + this.ShapePartOffsetMeans[this.ShapePartRange][this.XYRange],
                    this.ShapePartOffsetPrecisions[this.ShapePartRange][this.XYRange]);

            const double LocationDamping = 0.0002;
            var shapePartLocationDamped = Variable
                .Array(Variable.Array(Variable.Array<double>(this.XYRange), this.ShapePartRange), this.ObservationRange)
                .Named("shape_part_location_damped");
            shapePartLocationDamped[this.ObservationRange][this.ShapePartRange][this.XYRange] = Variable<double>.Factor(
                Damp.Backward<double>, this.ShapePartLocation[this.ObservationRange][this.ShapePartRange][this.XYRange], LocationDamping);

            const double OrientationPriorShape = 2.01;
            const double MeanEllipseHalfSize = 0.2;
            this.ShapePartOrientationPrior = Variable.Array<Wishart>(this.ShapePartRange).Named("shape_part_orientation_prior");
            this.ShapePartOrientationPrior.ObservedValue = Util.ArrayInit(
                shapePartCount,
                i => Wishart.FromShapeAndRate(
                    OrientationPriorShape, PositiveDefiniteMatrix.IdentityScaledBy(2, OrientationPriorShape * (MeanEllipseHalfSize * MeanEllipseHalfSize))));
            
            this.ShapePartOrientation = Variable.Array(Variable.Array<PositiveDefiniteMatrix>(this.ShapePartRange), this.ObservationRange).Named("shape_part_orientation");
            this.ShapePartOrientation[this.ObservationRange][this.ShapePartRange] = Variable<PositiveDefiniteMatrix>.Random(
                this.ShapePartOrientationPrior[this.ShapePartRange]).ForEach(this.ObservationRange);

            this.ShapePartConfigurationsCombined = Variable.Array(Variable.Array<Tuple<PositiveDefiniteMatrix, Vector>>(this.ShapePartRange), this.ObservationRange).Named("shape_part_configurations");
            this.ShapePartConfigurationsCombined[this.ObservationRange][this.ShapePartRange] = Variable<Tuple<PositiveDefiniteMatrix, Vector>>.Factor(
                ShapeFactors.Combine,
                shapePartLocationDamped[this.ObservationRange][this.ShapePartRange][0],
                shapePartLocationDamped[this.ObservationRange][this.ShapePartRange][1],
                this.ShapePartOrientation[this.ObservationRange][this.ShapePartRange]);
            this.ShapePartConfigurationsCombined[this.ObservationRange][this.ShapePartRange].AddAttribute(new MarginalPrototype(new VectorGaussianWishart(2)));

            // Brake shape part symmetry
            this.ShapePartLocation.InitialiseTo(
                Distribution<double>.Array(Util.ArrayInit(observationCount, i => Util.ArrayInit(shapePartCount, j => Util.ArrayInit(2, k => Gaussian.FromMeanAndVariance(Rand.Double(), 1.0 * 1.0))))));
            // Brake shape part symmetry and fix convergence issues at the same time
            //this.ShapePartConfigurationsCombined.InitialiseTo(
            //    Distribution<Tuple<PositiveDefiniteMatrix, Vector>>.Array(
            //        Util.ArrayInit(observationCount, i =>
            //            Util.ArrayInit(shapePartCount, j =>
            //                new VectorGaussianWishart(this.ShapePartOrientationPrior.ObservedValue[j].Shape, this.ShapePartOrientationPrior.ObservedValue[j].Rate, Vector.FromArray(0.4 + Rand.Double() * 0.2, 0.4 + Rand.Double() * 0.2), 0.01)))));

            // Break offset symmetry
            //var shapePartOffsetMeansTranspose =
            //    Variable.Array(Variable.Array<double>(this.ShapePartRange), this.XYRange).Named("shape_part_mean_offset_transposed");
            //shapePartOffsetMeansTranspose[this.XYRange][this.ShapePartRange] =
            //    Variable.Copy(this.ShapePartOffsetMeans[this.ShapePartRange][this.XYRange]);
            //Variable.ConstrainEqual(Variable.Sum(shapePartOffsetMeansTranspose[this.XYRange]), 0);

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

            using (Variable.ForEach(this.ObservationRange))
            using (Variable.ForEach(this.WidthRange))
            using (Variable.ForEach(this.HeightRange))
            {
                var labelsByPart = Variable.Array<bool>(this.ShapePartRange).Named("labels_by_part");

                using (Variable.ForEach(this.ShapePartRange))
                {
                    const double ShapePartConfigurationDamping = 0.001;
                    Variable<Tuple<PositiveDefiniteMatrix, Vector>> dampedShapePartConfiguration = Variable<Tuple<PositiveDefiniteMatrix, Vector>>.Factor(
                        Damp.Backward<Tuple<PositiveDefiniteMatrix, Vector>>,
                        this.ShapePartConfigurationsCombined[this.ObservationRange][this.ShapePartRange],
                        ShapePartConfigurationDamping).Named("shape_part_combined_damped");
                    labelsByPart[this.ShapePartRange] = Variable<bool>.Factor(
                        ShapeFactors.LabelFromShape,
                        this.PixelCoords[this.WidthRange, this.HeightRange],
                        dampedShapePartConfiguration);
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

        public bool[][,] Sample(int sampleCount)
        {
            bool[][,] result = new bool[sampleCount][,];
            for (int sample = 0; sample < sampleCount; ++sample)
            {
                // Sample locations from priors
                double locationX = this.ShapeLocationPrior.ObservedValue[0].Sample();
                double locationY = this.ShapeLocationPrior.ObservedValue[1].Sample();

                // Sample shape part locations
                List<Vector> shapePartLocations = new List<Vector>();
                List<PositiveDefiniteMatrix> shapePartOrientations = new List<PositiveDefiniteMatrix>();

                for (int i = 0; i < this.ShapePartRange.SizeAsInt; ++i)
                {
                    double x = locationX +
                        Gaussian.FromMeanAndPrecision(this.ShapePartOffsetMeans.ObservedValue[i][0], this.ShapePartOffsetPrecisions.ObservedValue[i][1]).Sample();
                    double y = locationY +
                        Gaussian.FromMeanAndPrecision(this.ShapePartOffsetMeans.ObservedValue[i][1], this.ShapePartOffsetPrecisions.ObservedValue[i][1]).Sample();
                    PositiveDefiniteMatrix orientation = this.ShapePartOrientationPrior.ObservedValue[i].Sample();

                    shapePartLocations.Add(Vector.FromArray(x, y));
                    shapePartOrientations.Add(orientation);
                }

                result[sample] = new bool[this.WidthRange.SizeAsInt, this.HeightRange.SizeAsInt];
                for (int i = 0; i < this.WidthRange.SizeAsInt; ++i)
                {
                    for (int j = 0; j < this.HeightRange.SizeAsInt; ++j)
                    {
                        bool label = false;
                        for (int k = 0; k < this.ShapePartRange.SizeAsInt; ++k)
                        {
                            label |= ShapeFactors.LabelFromShape(
                                this.PixelCoords.ObservedValue[i, j],
                                Tuple.Create(shapePartOrientations[k], shapePartLocations[k]));
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
                this.ShapePartConfigurationsCombined,
                this.ShapePartOffsetMeans,
                this.ShapePartOffsetPrecisions
            };
        }
    }
}