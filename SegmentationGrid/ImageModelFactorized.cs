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
        public Range WidthRange { get; private set; }

        public Range HeightRange { get; private set; }

        public Range ObservationRange { get; private set; }

        // Shape model

        public Variable<Beta>[] ShapePartPresenseProbabilityPrior { get; private set; }

        public Variable<double>[] ShapePartPresenseProbability { get; private set; }

        public VariableArray<bool>[] ShapePartPresented { get; private set; }

        public Variable<Gaussian>[] ShapeLocationPrior { get; private set; }

        public VariableArray<double>[] ShapeLocation { get; private set; }

        public VariableArray<double>[][] ShapePartLocation { get; private set; }

        public Variable<PositiveDefiniteMatrix>[] ShapePartOrientation { get; private set; }

        public Variable<Gaussian>[][] ShapePartOffsetMeanPriors { get; private set; }

        public Variable<Gamma>[][] ShapePartOffsetPrecisionPriors { get; private set; }

        public Variable<double>[][] ShapePartOffsetMeans { get; private set; }

        public Variable<double>[][] ShapePartOffsetPrecisions { get; private set; }

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
            this.WidthRange = new Range(width).Named("width_range");
            this.HeightRange = new Range(height).Named("height_range");
            this.ObservationRange = new Range(observationCount).Named("observation_range");

            //this.ObservationRange.AddAttribute(new Sequential());
            //this.WidthRange.AddAttribute(new Sequential());
            //this.HeightRange.AddAttribute(new Sequential());

            this.ShapeLocationPrior = Util.ArrayInit(
                2, j => Variable.Observed(Gaussian.FromMeanAndVariance(0.5, 0.5 * 0.5)).Named(string.Format("shape_location_prior_{0}", j)));
            this.ShapeLocation = Util.ArrayInit(
                2, j => Variable.Array<double>(this.ObservationRange).Named(string.Format("shape_location_{0}", j)));
            for (int j = 0; j < 2; ++j)
            {
                this.ShapeLocation[j][this.ObservationRange] =
                    Variable<double>.Random(this.ShapeLocationPrior[j]).ForEach(this.ObservationRange);
            }

            ShapePartPresenseProbabilityPrior = Util.ArrayInit(
                shapePartCount, i => Variable.Observed(new Beta(1, 1)).Named(string.Format("shape_part_presense_prob_prior_{0}", i)));
            this.ShapePartPresenseProbability = new Variable<double>[shapePartCount];
            for (int i = 0; i < shapePartCount; ++i)
            {
                this.ShapePartPresenseProbability[i] =
                    Variable<double>.Random(this.ShapePartPresenseProbabilityPrior[i])
                                    .Named(string.Format("shape_part_presense_prob_{0}", i));
            }

            this.ShapePartOffsetMeanPriors = new Variable<Gaussian>[shapePartCount][];
            this.ShapePartOffsetPrecisionPriors = new Variable<Gamma>[shapePartCount][];
            this.ShapePartOffsetMeans = new Variable<double>[shapePartCount][];
            this.ShapePartOffsetPrecisions = new Variable<double>[shapePartCount][];

            this.ShapePartPresented = new VariableArray<bool>[shapePartCount];
            this.ShapePartLocation = new VariableArray<double>[shapePartCount][];
            this.ShapePartOrientation = new Variable<PositiveDefiniteMatrix>[shapePartCount];

            for (int i = 0; i < shapePartCount; ++i)
            {
                this.ShapePartOffsetMeanPriors[i] = Util.ArrayInit(
                    2, j => Variable.Observed(Gaussian.FromMeanAndVariance(0.0, 0.5 * 0.5)).Named(string.Format("shape_offset_mean_prior_{0}_{1}", i, j)));
                this.ShapePartOffsetPrecisionPriors[i] = Util.ArrayInit(
                    2, j => Variable.Observed(Gamma.FromMeanAndVariance(1.0 / (0.05 * 0.05), 400 * 400)).Named(string.Format("shape_offset_precision_prior_{0}_{1}", i, j)));
                this.ShapePartOffsetMeans[i] = Util.ArrayInit(
                    2, j => Variable<double>.Random(this.ShapePartOffsetMeanPriors[i][j]).Named(string.Format("shape_offset_mean_{0}_{1}", i, j)));
                this.ShapePartOffsetPrecisions[i] = Util.ArrayInit(
                    2, j => Variable<double>.Random(this.ShapePartOffsetPrecisionPriors[i][j]).Named(string.Format("shape_offset_precision_{0}_{1}", i, j)));

                this.ShapePartPresented[i] = Variable.Array<bool>(this.ObservationRange).Named(string.Format("shape_part_presense_{0}", i));
                this.ShapePartPresented[i][this.ObservationRange] =
                    Variable.Bernoulli(this.ShapePartPresenseProbability[i]).ForEach(this.ObservationRange);

                this.ShapePartLocation[i] = Util.ArrayInit(
                    2, j => Variable.Array<double>(this.ObservationRange).Named(string.Format("shape_location_{0}_{1}", i, j)));
                for (int j = 0; j < 2; ++j)
                {
                    using (Variable.ForEach(this.ObservationRange))
                    {
                        // No damping
                        Variable<double> offset = Variable
                            .GaussianFromMeanAndPrecision(this.ShapePartOffsetMeans[i][j], this.ShapePartOffsetPrecisions[i][j])
                            .Named(string.Format("shape_offset_{0}_{1}", i, j));
                        this.ShapePartLocation[i][j][this.ObservationRange] = this.ShapeLocation[j][this.ObservationRange] + offset;

                        //const double damping = 1.0;

                        // Damp mean and precision
                        //Variable<double> dampedOffsetMean =
                        //    Variable<double>.Factor(Damp.Backward<double>, this.ShapeOffsetMeans[i - 1][j], damping);
                        //Variable<double> dampedOffsetPrecision =
                        //    Variable<double>.Factor(Damp.Backward<double>, this.ShapeOffsetPrecisions[i - 1][j], damping);
                        //Variable<double> offset = Variable.GaussianFromMeanAndPrecision(dampedOffsetMean, dampedOffsetPrecision).Named(string.Format("shape_offset_{0}_{1}", i, j));
                        //this.ShapeLocation[i][j][this.ObservationRange] = this.ShapeLocation[i - 1][j][this.ObservationRange] + offset;

                        // Damp offset
                        //Variable<double> offset = Variable
                        //    .GaussianFromMeanAndPrecision(this.ShapeOffsetMeans[i - 1][j], this.ShapeOffsetPrecisions[i - 1][j])
                        //    .Named(string.Format("shape_offset_{0}_{1}", i, j));
                        //Variable<double> dampedOffset = Variable<double>
                        //    .Factor(Damp.Backward<double>, offset, damping)
                        //    .Named(string.Format("damped_shape_offset_{0}_{1}", i, j));
                        //this.ShapeLocation[i][j][this.ObservationRange] = this.ShapeLocation[i - 1][j][this.ObservationRange] + dampedOffset;
                    }
                }

                this.ShapePartOrientation[i] = Variable.New<PositiveDefiniteMatrix>().Named(string.Format("shape_orientation_{0}", i));
            }

            this.ObservationNoiseProbabilityPrior = Variable.New<Beta>().Named("observation_noise_prob_prior");
            this.ObservationNoiseProbabilityPrior.ObservedValue = new Beta(1, 50);
            this.ObservationNoiseProbability = Variable.Random<double, Beta>(this.ObservationNoiseProbabilityPrior).Named("observation_noise_prob");

            this.PottsPenalty = Variable.New<double>();
            this.PottsPenalty.ObservedValue = 0.9;

            this.PixelCoords = Variable.Array<Vector>(this.WidthRange, this.HeightRange).Named("pixel_coords");
            this.PixelCoords.ObservedValue = Util.ArrayInit(width, height, (i, j) => Vector.FromArray((i + 0.5) / width, (j + 0.5) / height));

            this.Labels = Variable.Array<VariableArray2D<bool>, bool[][,]>(Variable.Array<bool>(this.WidthRange, this.HeightRange), this.ObservationRange).Named("labels");
            this.ObservedLabels = Variable.Array<VariableArray2D<bool>, bool[][,]>(Variable.Array<bool>(this.WidthRange, this.HeightRange), this.ObservationRange).Named("observed_labels");

            using (Variable.ForEach(this.ObservationRange))
            {
                VariableArray2D<bool> anyLabel = null;

                for (int i = 0; i < shapePartCount; ++i)
                {
                    VariableArray2D<bool> partLabel = Variable.Array<bool>(this.WidthRange, this.HeightRange).Named(string.Format("part_label_{0}", i));

                    using (Variable.If(this.ShapePartPresented[i][this.ObservationRange]))
                    {
                        using (Variable.ForEach(this.WidthRange))
                        using (Variable.ForEach(this.HeightRange))
                        {
                            const double damping = 0.01;
                            Variable<double> dampedShapeLocationX =
                                Variable<double>.Factor(Damp.Backward<double>, this.ShapePartLocation[i][0][this.ObservationRange], damping).Named(string.Format("damped_shape_location_{0}_x", i));
                            Variable<double> dampedShapeLocationY =
                                Variable<double>.Factor(Damp.Backward<double>, this.ShapePartLocation[i][1][this.ObservationRange], damping).Named(string.Format("damped_shape_location_{0}_y", i));

                            partLabel[this.WidthRange, this.HeightRange] = Variable<bool>.Factor(
                                ShapeFactors.LabelFromShape,
                                this.PixelCoords[this.WidthRange, this.HeightRange],
                                dampedShapeLocationX,
                                dampedShapeLocationY,
                                //this.ShapeLocation[i][0],
                                //this.ShapeLocation[i][1],
                                this.ShapePartOrientation[i]);                            
                        }
                    }

                    using (Variable.IfNot(this.ShapePartPresented[i][this.ObservationRange]))
                    {
                        partLabel[this.WidthRange, this.HeightRange] = false;
                    }

                    if (ReferenceEquals(anyLabel, null))
                    {
                        anyLabel = partLabel;
                    }
                    else
                    {
                        var anyLabelNew = Variable.Array<bool>(this.WidthRange, this.HeightRange);
                        anyLabelNew[this.WidthRange, this.HeightRange] =
                            anyLabel[this.WidthRange, this.HeightRange] | partLabel[this.WidthRange, this.HeightRange];
                        anyLabel = anyLabelNew;
                    }
                }

                this.Labels[this.ObservationRange][this.WidthRange, this.HeightRange] =
                    anyLabel[this.WidthRange, this.HeightRange]; // TODO: copy here?

                using (var widthIterationBlock = Variable.ForEach(this.WidthRange))
                using (var heightIterationBlock = Variable.ForEach(this.HeightRange))
                {
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
        }

        public bool[,] Sample()
        {
            // Sample locations from priors
            double locationX = this.ShapeLocationPrior[0].ObservedValue.Sample();
            double locationY = this.ShapeLocationPrior[1].ObservedValue.Sample();

            // Sample shape part locations
            List<double> shapePartLocationX = new List<double>();
            List<double> shapePartLocationY = new List<double>();
            List<bool> shapePartPresense = new List<bool>();
            for (int i = 0; i < this.ShapePartLocation.Length; ++i)
            {
                double x = locationX +
                    Gaussian.FromMeanAndPrecision(this.ShapePartOffsetMeans[i][0].ObservedValue, this.ShapePartOffsetPrecisions[i][0].ObservedValue).Sample();
                double y = locationY +
                    Gaussian.FromMeanAndPrecision(this.ShapePartOffsetMeans[i][1].ObservedValue, this.ShapePartOffsetPrecisions[i][1].ObservedValue).Sample();
                bool presense = Rand.Double() < this.ShapePartPresenseProbability[i].ObservedValue;

                shapePartLocationX.Add(x);
                shapePartLocationY.Add(y);
                shapePartPresense.Add(presense);
            }

            bool[,] result = new bool[this.WidthRange.SizeAsInt, this.HeightRange.SizeAsInt];
            for (int i = 0; i < this.WidthRange.SizeAsInt; ++i)
            {
                for (int j = 0; j < this.HeightRange.SizeAsInt; ++j)
                {
                    bool label = false;
                    for (int k = 0; k < this.ShapePartLocation.Length; ++k)
                    {
                        if (shapePartPresense[k])
                        {
                            label |= ShapeFactors.LabelFromShape(
                                this.PixelCoords.ObservedValue[i, j],
                                shapePartLocationX[k],
                                shapePartLocationY[k],
                                this.ShapePartOrientation[k].ObservedValue);
                        }
                    }

                    if (Rand.Double() < this.ObservationNoiseProbability.ObservedValue)
                    {
                        label = !label;
                    }

                    result[i, j] = label;
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
            List<IVariable> vars = new List<IVariable>();
            for (int i = 0; i < this.ShapePartLocation.Length; ++i)
            {
                vars.Add(this.ShapePartLocation[i][0]);
                vars.Add(this.ShapePartLocation[i][1]);
            }

            return vars.ToArray();
        }

        public IVariable[] GetVariablesForShapeModelLearning()
        {
            throw new NotImplementedException();
        }
    }
}