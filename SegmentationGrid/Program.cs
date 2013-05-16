using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Utils;

namespace SegmentationGrid
{
    using BernoulliArray1D = DistributionStructArray<Bernoulli, bool>;
    using GaussianArray2D = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using BernoulliArray2D = DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>;
    using GammaArray2D = DistributionRefArray<DistributionStructArray<Gamma, double>, double[]>;
    using GaussianArray3D = DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>, double[][]>;

    class Program
    {
        const int GridSize = 120;

        private static void Main()
        {
            Rand.Restart(666);

            const double observationNoiseProb = 0.01;
            const double pottsPenalty = 0.9;
            const int trainingSetSize = 20;

            // 1 ellipse
            //var offsetMeans = new Vector[0];
            //var offsetPrecisions = new Vector[0];
            //var shapeOrientations = new[] { EllipsePrecisionMatrix(0.2, 0.05, 0) };
            //var shapeLocationPrior = new[] { Gaussian.FromMeanAndVariance(0.5, 0.2 * 0.2), Gaussian.FromMeanAndVariance(0.5, 0.3 * 0.3) };

            // 2 ellipses
            //var offsetMeans = new[] { Vector.FromArray(0.0, 0.2) };
            //var offsetPrecisions = new[] { Vector.FromArray(1.0 / (0.3 * 0.3), 1.0 / (0.1 * 0.1)) };
            //var shapeOrientations = new[] { EllipsePrecisionMatrix(0.2, 0.05, 0), EllipsePrecisionMatrix(0.1, 0.3, 0) };
            //var shapeLocationPrior = new[] { Gaussian.FromMeanAndVariance(0.5, 0.2 * 0.2), Gaussian.FromMeanAndVariance(0.5, 0.2 * 0.2) };

            // 3 ellipses
            var offsetMeans = new[] { Vector.FromArray(0.0, 0.0), Vector.FromArray(0.0, -0.2), Vector.FromArray(0.0, 0.2) };
            var offsetPrecisions = new[] { Vector.FromArray(1.0 / (0.01 * 0.01), 1.0 / (0.01 * 0.01)), Vector.FromArray(1.0 / (0.01 * 0.01), 1.0 / (0.01 * 0.01)), Vector.FromArray(1.0 / (0.1 * 0.1), 1.0 / (0.01 * 0.01)) };
            var shapeOrientations = new[] { EllipsePrecisionMatrix(0.1, 0.2, 0), EllipsePrecisionMatrix(0.02, 0.15, 0), EllipsePrecisionMatrix(0.2, 0.02, 0) };
            var shapeLocationPrior = new[] { Gaussian.FromMeanAndVariance(0.5, 0.1 * 0.1), Gaussian.FromMeanAndVariance(0.5, 0.1 * 0.1) };

            // Sample labels from true model

            bool[][,] sampledLabels = SampleNoisyLabels(
                shapeOrientations, offsetMeans, offsetPrecisions, shapeLocationPrior, observationNoiseProb, pottsPenalty, trainingSetSize);

            // Save samples
            for (int i = 0; i < sampledLabels.Length; ++i)
            {
                ImageHelpers.ArrayToBitmap(sampledLabels[i], b => b ? Color.Red : Color.Green).Save(string.Format("sampled_labels_{0}.png", i));
            }

            var shapeOrientationsToFit = new List<PositiveDefiniteMatrix>();

            // Add slightly modified initial shapes
            for (int i = 0; i < shapeOrientations.Count(); ++i)
            {
                const double ScaleNoiseAmplitude = 0.0;
                const double AngleNoiseAmplitude = 0.0;
                
                double stdDev1, stdDev2, angle;
                ExtractScaleAndAngle(shapeOrientations[i], out stdDev1, out stdDev2, out angle);
                stdDev1 = Math.Max(0.001, stdDev1 + (Rand.Double() * 2 - 1) * ScaleNoiseAmplitude);
                stdDev2 = Math.Max(0.001, stdDev2 + (Rand.Double() * 2 - 1) * ScaleNoiseAmplitude);
                angle += (Rand.Double() * 2 - 1) * Math.PI * AngleNoiseAmplitude;
                shapeOrientationsToFit.Add(EllipsePrecisionMatrix(stdDev1, stdDev2, angle));
            }
            
            // Add redundant shapes
            const int RedundantShapeCount = 7;
            for (int i = 0; i < RedundantShapeCount; ++i)
            {
                double randomWidth = 0.02 + 0.2 * Rand.Double();
                double randomHeight = 0.02 + 0.2 * Rand.Double();
                double randomAngle = Math.PI * 2 * Rand.Double();

                PositiveDefiniteMatrix shapeOrientation = EllipsePrecisionMatrix(randomWidth, randomHeight, randomAngle);

                // TODO: remove condition
                if (i != 3)
                {
                    shapeOrientationsToFit.Add(shapeOrientation);
                }
            }

            LearnShapeModel(shapeOrientationsToFit.ToArray(), sampledLabels, observationNoiseProb, pottsPenalty);
        }

        private static bool[][,] SampleNoisyLabels(
            PositiveDefiniteMatrix[] shapeOrientations,
            Vector[] offsetMeans,
            Vector[] offsetPrecisions,
            Gaussian[] shapeLocationPrior,
            double observationNoiseProb,
            double pottsPenalty,
            int count)
        {
            int shapePartCount = shapeOrientations.Length;

            //var gridHolder = new ImageModelFullCovariance(GridSize, GridSize, shapeParams.Length);
            var gridHolder = new ImageModelFactorized(GridSize, GridSize, shapePartCount, 1);

            // Specify sampling model
            gridHolder.ShapeLocationPrior.ObservedValue = shapeLocationPrior;
            gridHolder.ShapePartOrientation.ObservedValue = shapeOrientations;
            gridHolder.ObservationNoiseProbability.ObservedValue = observationNoiseProb;
            gridHolder.PottsPenalty.ObservedValue = pottsPenalty;
            gridHolder.ShapePartOffsetMeans.ObservedValue = Util.ArrayInit(shapePartCount, i => offsetMeans[i].ToArray());
            gridHolder.ShapePartOffsetPrecisions.ObservedValue = Util.ArrayInit(shapePartCount, i => offsetPrecisions[i].ToArray());
            gridHolder.ShapePartPresentedPrior.ObservedValue = Util.ArrayInit(shapePartCount, i => new Bernoulli(1));

            return gridHolder.Sample(count);
        }

        private static void LearnShapeModel(
            PositiveDefiniteMatrix[] shapeOrientations,
            bool[][,] trainingSet,
            double observationNoiseProb,
            double pottsPenalty)
        {
            int observationCount = trainingSet.Length;
            int shapePartCount = shapeOrientations.Length;

            var gridHolder = new ImageModelFactorized(GridSize, GridSize, shapePartCount, observationCount);
            gridHolder.PottsPenalty.ObservedValue = pottsPenalty;
            gridHolder.ObservationNoiseProbability.ObservedValue = observationNoiseProb;
            gridHolder.ObservedLabels.ObservedValue = trainingSet;
            gridHolder.ShapePartOrientation.ObservedValue = shapeOrientations;

            InferenceEngine engine = CreateInferenceEngine();
            engine.OptimiseForVariables = gridHolder.GetVariablesForShapeModelLearning();

            // Initialize messages randomly
            //Gaussian[][][] randomLocationInitialization = Util.ArrayInit(
            //    observationCount, i => Util.ArrayInit(
            //        shapePartCount,
            //        j => Util.ArrayInit(2, k => Gaussian.FromMeanAndVariance(0.5 + 0.25 * (2 * Rand.Double() - 1), 0.5 * 0.5))));
            //gridHolder.ShapePartLocation.InitialiseTo(Distribution<double>.Array(randomLocationInitialization));

            for (int iterationCount = 10; iterationCount <= 500; iterationCount += 10)
            {
                engine.NumberOfIterations = iterationCount;

                // Infer locations
                GaussianArray3D shapePartLocationPosteriors = engine.Infer<GaussianArray3D>(gridHolder.ShapePartLocation);
                BernoulliArray1D shapePartPresentedPosteriors = engine.Infer<BernoulliArray1D>(gridHolder.ShapePartPresented);
                
                // Draw part locations
                for (int i = 0; i < trainingSet.Length; ++i)
                {
                    var shapeParams = Util.ArrayInit(
                        shapePartCount,
                        j => Tuple.Create(
                            shapeOrientations[j],
                            Vector.FromArray(shapePartLocationPosteriors[i][j][0].GetMean(), shapePartLocationPosteriors[i][j][1].GetMean())));
                    shapeParams = shapeParams.Where((s, j) => shapePartPresentedPosteriors[j].GetProbTrue() > 0.5).ToArray();
                    DrawShape(shapeParams).Save(string.Format("fitted_object_{0}_{1}.png", iterationCount, i));
                }

                // Infer shape model
                GaussianArray2D shapeOffsetMeanPosteriors = engine.Infer<GaussianArray2D>(gridHolder.ShapePartOffsetMeans);
                GammaArray2D shapeOffsetPrecPosteriors = engine.Infer<GammaArray2D>(gridHolder.ShapePartOffsetPrecisions);

                // Print shape model
                for (int i = 0; i < shapePartCount; ++i)
                {
                    Console.WriteLine("Shape part {0}", i);
                    Console.WriteLine("Offset mean X: {0}", shapeOffsetMeanPosteriors[i][0]);
                    Console.WriteLine("Offset mean Y: {0}", shapeOffsetMeanPosteriors[i][1]);
                    Console.WriteLine("Offset prec X: {0}", shapeOffsetPrecPosteriors[i][0]);
                    Console.WriteLine("Offset prec Y: {0}", shapeOffsetPrecPosteriors[i][1]);
                    Console.WriteLine("Presense: {0}", shapePartPresentedPosteriors[i]);
                    Console.WriteLine();
                }
            }
        }

        private static PositiveDefiniteMatrix EllipsePrecisionMatrix(
            double stdDev1, double stdDev2, double angle)
        {
            PositiveDefiniteMatrix precisionNoRotation = new PositiveDefiniteMatrix(2, 2);
            precisionNoRotation[0, 0] = 1.0 / (stdDev1 * stdDev1);
            precisionNoRotation[1, 1] = 1.0 / (stdDev2 * stdDev2);

            PositiveDefiniteMatrix rotationMatrix = RotationMatrix(angle);
            PositiveDefiniteMatrix rotationMatrixInverse = RotationMatrix(-angle);

            PositiveDefiniteMatrix rotatedPrecision = new PositiveDefiniteMatrix(2, 2);
            rotatedPrecision.SetTo(rotationMatrixInverse * precisionNoRotation * rotationMatrix);

            return rotatedPrecision;
        }

        private static void ExtractScaleAndAngle(
            PositiveDefiniteMatrix precisionMatrix, out double stdDev1, out double stdDev2, out double angle)
        {
            PositiveDefiniteMatrix rotation = new PositiveDefiniteMatrix(2, 2);
            rotation.SetToEigenvectorsOfSymmetric(precisionMatrix);
            Matrix precisionNoRotation = rotation * precisionMatrix * rotation.Inverse();

            stdDev1 = 1.0 / Math.Sqrt(precisionNoRotation[0, 0]);
            stdDev2 = 1.0 / Math.Sqrt(precisionNoRotation[1, 1]);
            angle = Math.Atan2(rotation[1, 0], rotation[0, 0]);
        }

        private static PositiveDefiniteMatrix RotationMatrix(double angle)
        {
            PositiveDefiniteMatrix result = new PositiveDefiniteMatrix(2, 2);
            result[0, 0] = Math.Cos(angle);
            result[0, 1] = -Math.Sin(angle);
            result[1, 0] = Math.Sin(angle);
            result[1, 1] = Math.Cos(angle);
            return result;
        }

        private static Bitmap DrawShape(Tuple<PositiveDefiniteMatrix, Vector>[] shapeParams)
        {
            double[,] mask = Util.ArrayInit(
                GridSize,
                GridSize,
                (x, y) =>
                {
                    double prob = 1.0;
                    for (int i = 0; i < shapeParams.Length; ++i)
                    {
                        Bernoulli partLabelDistr = LabelFromShapeFactorizedOps.LabelAverageConditional(
                            Vector.FromArray((x + 0.5) / GridSize, (y + 0.5) / GridSize),
                            shapeParams[i].Item2[0],
                            shapeParams[i].Item2[1],
                            shapeParams[i].Item1);
                        prob *= (1 - partLabelDistr.GetProbTrue());
                    }

                    return 1 - prob;
                });

            Bitmap result = ImageHelpers.ArrayToBitmap(mask, p => PenaltyToColor(p, 0, 1));
            return result;
        }

        private static InferenceEngine CreateInferenceEngine()
        {
            var engine = new InferenceEngine();
            engine.NumberOfIterations = 10;
            //engine.ShowSchedule = true;
            //engine.ShowProgress = false;
            engine.Compiler.RecommendedQuality = QualityBand.Unknown;
            engine.Compiler.RequiredQuality = QualityBand.Unknown;
            engine.Compiler.GenerateInMemory = false;
            engine.Compiler.WriteSourceFiles = true;
            engine.Compiler.IncludeDebugInformation = true;
            //engine.Compiler.UseSerialSchedules = true;
            return engine;
        }

        private static Color RgbLabelToColor(int label)
        {
            switch (label)
            {
                case 0:
                    return Color.Red;
                case 1:
                    return Color.Green;
                case 2:
                    return Color.Blue;
                default:
                    Debug.Fail("Unknown label!");
                    return Color.Black;
            }
        }

        private static Color PenaltyToColor(double penalty, double minPenalty, double maxPenalty)
        {
            penalty = Math.Max(minPenalty, Math.Min(maxPenalty, penalty));
            double averagePenalty = 0.5 * (maxPenalty + minPenalty);
            double relativeDeviation = 2 * (penalty - averagePenalty) / (maxPenalty - minPenalty);
            Debug.Assert(relativeDeviation >= -1 && relativeDeviation <= 1);
            if (Math.Abs(relativeDeviation) < 0.01) return Color.White; // Draw white zero-level
            return Color.FromArgb((int)(Math.Max(relativeDeviation, 0) * 255), 0, (int)(Math.Max(-relativeDeviation, 0) * 255));
        }
    }
}
