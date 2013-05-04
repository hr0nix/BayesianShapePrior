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
        public Range WidthRange { get; private set; }

        public Range HeightRange { get; private set; }

        // Shape model

        public Variable<VectorGaussianWishart>[] ShapeParamsPrior { get; private set; }

        public Variable<Tuple<PositiveDefiniteMatrix, Vector>>[] ShapeParams { get; private set; }

        // Observation noise model

        public Variable<Beta> ObservationNoiseProbabilityPrior { get; private set; }

        public Variable<double> ObservationNoiseProbability { get; private set; }

        // Grid specification

        public VariableArray2D<Vector> PixelCoords { get; private set; }

        public VariableArray2D<bool> Labels { get; private set; }

        public VariableArray2D<bool> ObservedLabels { get; private set; }

        public Variable<double> PottsPenalty { get; private set; }

        public ImageModelFullCovariance(int width, int height, int shapePartCount)
        {
            this.WidthRange = new Range(width).Named("width_range");
            this.HeightRange = new Range(height).Named("height_range");

            this.ShapeParamsPrior = new Variable<VectorGaussianWishart>[shapePartCount];
            this.ShapeParams = new Variable<Tuple<PositiveDefiniteMatrix, Vector>>[shapePartCount];
            for (int i = 0; i < shapePartCount; ++i)
            {
                this.ShapeParamsPrior[i] = Variable.New<VectorGaussianWishart>().Named(string.Format("shape_prior_{0}", i));
                this.ShapeParams[i] = Variable.Random<Tuple<PositiveDefiniteMatrix, Vector>, VectorGaussianWishart>(
                    this.ShapeParamsPrior[i]).Named(string.Format("shape_{0}", i));

                // Initialize priors
                const double firstShape = 3.0;
                const double meanEllipseHalfSize = 0.15;
                this.ShapeParamsPrior[i].ObservedValue = new VectorGaussianWishart(
                    firstShape,
                    PositiveDefiniteMatrix.IdentityScaledBy(2, firstShape * (meanEllipseHalfSize * meanEllipseHalfSize)),
                    Vector.FromArray(0.5, 0.5),
                    1.0);
            }

            this.ObservationNoiseProbabilityPrior = Variable.New<Beta>().Named("observation_noise_prob_prior");
            this.ObservationNoiseProbabilityPrior.ObservedValue = new Beta(1, 50);
            this.ObservationNoiseProbability = Variable.Random<double, Beta>(this.ObservationNoiseProbabilityPrior).Named("observation_noise_prob");

            this.PottsPenalty = Variable.New<double>();
            this.PottsPenalty.ObservedValue = 0.9;

            this.PixelCoords = Variable.Array<Vector>(this.WidthRange, this.HeightRange).Named("pixel_coords");
            this.PixelCoords.ObservedValue = Util.ArrayInit(width, height, (i, j) => Vector.FromArray((i + 0.5) / width, (j + 0.5) / height));

            this.Labels = Variable.Array<bool>(this.WidthRange, this.HeightRange).Named("labels");
            this.ObservedLabels = Variable.Array<bool>(this.WidthRange, this.HeightRange).Named("observed_labels");
            using (var widthIterationBlock = Variable.ForEach(this.WidthRange))
            using (var heightIterationBlock = Variable.ForEach(this.HeightRange))
            {
                Variable<bool> anyLabel = null;
                for (int i = 0; i < shapePartCount; ++i)
                {
                    const double damping = 0.003;
                    Variable<Tuple<PositiveDefiniteMatrix, Vector>> dampedShapeParams = Variable<Tuple<PositiveDefiniteMatrix, Vector>>.Factor(
                        Damp.Backward<Tuple<PositiveDefiniteMatrix, Vector>>,
                        this.ShapeParams[i],
                        damping).Named(string.Format("damped_shape_{0}", i));
                    
                    Variable<bool> partLabel = Variable<bool>.Factor(
                        ShapeFactors.LabelFromShape,
                        this.PixelCoords[this.WidthRange, this.HeightRange],
                        dampedShapeParams).Named(string.Format("part_label_{0}", i));
                    anyLabel = object.ReferenceEquals(anyLabel, null) ? partLabel : partLabel | anyLabel;
                }

                this.Labels[this.WidthRange, this.HeightRange] = anyLabel;
                
                using (Variable.If(this.Labels[this.WidthRange, this.HeightRange]))
                {
                    this.ObservedLabels[this.WidthRange, this.HeightRange] = !Variable.Bernoulli(this.ObservationNoiseProbability);
                }
                using (Variable.IfNot(this.Labels[this.WidthRange, this.HeightRange]))
                {
                    this.ObservedLabels[this.WidthRange, this.HeightRange] = Variable.Bernoulli(this.ObservationNoiseProbability);
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

        public IVariable[] GetVariablesForSegmentation()
        {
            return new IVariable[] { this.Labels };
        }

        public IVariable[] GetVariablesForShapeDetection()
        {
            List<IVariable> vars = new List<IVariable>();
            for (int i = 0; i < this.ShapeParams.Length; ++i)
            {
                vars.Add(this.ShapeParams[i]);
            }

            return vars.ToArray();
        }

        public IVariable[] GetVariablesForShapeModelLearning()
        {
            throw new NotImplementedException();
        }
    }
}