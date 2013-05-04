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
    class Program
    {
        const int GridSize = 120;

        //private static void SampleLabels(
        //    Tuple<PositiveDefiniteMatrix, Vector>[] shapeParams,
        //    double observationNoiseProb,
        //    double pottsPenalty,
        //    out bool[,] labels,
        //    out bool[,] noisyLabels)
        //{
        //    //var gridHolder = new ImageModelFullCovariance(GridSize, GridSize, shapeParams.Length);
        //    var gridHolder = new ImageModelFactorized(GridSize, GridSize, shapeParams.Length);

        //    // Specify sampling model
        //    gridHolder.ObservationNoiseProbability.ObservedValue = observationNoiseProb;
        //    gridHolder.PottsPenalty.ObservedValue = pottsPenalty;
        //    for (int i = 0; i < shapeParams.Length; ++i)
        //    {
        //        //gridHolder.ShapeParams[i].ObservedValue = shapeParams[i];
        //        gridHolder.ShapeOrientation[i].ObservedValue = shapeParams[i].Item1;
        //        gridHolder.ShapeLocation[i][0].ObservedValue = shapeParams[i].Item2[0];
        //        gridHolder.ShapeLocation[i][1].ObservedValue = shapeParams[i].Item2[1];

        //        // TODO: to make gibbs sampling work
        //        if (i < shapeParams.Length - 1)
        //        {
        //            gridHolder.ShapeOffsetMeans[i][0].ObservedValue = 1;
        //            gridHolder.ShapeOffsetMeans[i][1].ObservedValue = 1;
        //            gridHolder.ShapeOffsetPrecisions[i][0].ObservedValue = 1;
        //            gridHolder.ShapeOffsetPrecisions[i][1].ObservedValue = 1;
        //        }
        //    }

        //    InferenceEngine samplingEngine = CreateInferenceEngine();
        //    var samplingAlgorithm = new GibbsSampling();
        //    samplingEngine.Algorithm = samplingAlgorithm;
        //    gridHolder.ObservedLabels.AddAttribute(QueryTypes.Samples);
        //    gridHolder.Labels.AddAttribute(QueryTypes.Samples);

        //    // Sample labels from model
        //    samplingAlgorithm.BurnIn = 50;
        //    samplingEngine.NumberOfIterations = samplingAlgorithm.BurnIn;
        //    labels = samplingEngine.Infer<IList<bool[,]>>(gridHolder.Labels, QueryTypes.Samples)[0];
        //    noisyLabels = samplingEngine.Infer<IList<bool[,]>>(gridHolder.ObservedLabels, QueryTypes.Samples)[0];
        //}

        private static bool[][,] SampleNoisyLabels(
            PositiveDefiniteMatrix[] shapeOrientations,
            Vector[] offsetMeans,
            Vector[] offsetPrecisions,
            double[] partPresenseProbs,
            Gaussian[] shapeLocationPrior,
            double observationNoiseProb,
            double pottsPenalty,
            int count)
        {
            //var gridHolder = new ImageModelFullCovariance(GridSize, GridSize, shapeParams.Length);
            var gridHolder = new ImageModelFactorized(GridSize, GridSize, shapeOrientations.Length, 1);

            // Specify sampling model
            gridHolder.ShapeLocationPrior[0].ObservedValue = shapeLocationPrior[0];
            gridHolder.ShapeLocationPrior[1].ObservedValue = shapeLocationPrior[1];
            gridHolder.ObservationNoiseProbability.ObservedValue = observationNoiseProb;
            gridHolder.PottsPenalty.ObservedValue = pottsPenalty;
            for (int i = 0; i < shapeOrientations.Length; ++i)
            {
                gridHolder.ShapePartOrientation[i].ObservedValue = shapeOrientations[i];
                gridHolder.ShapePartOffsetMeans[i][0].ObservedValue = offsetMeans[i][0];
                gridHolder.ShapePartOffsetMeans[i][1].ObservedValue = offsetMeans[i][1];
                gridHolder.ShapePartOffsetPrecisions[i][0].ObservedValue = offsetPrecisions[i][0];
                gridHolder.ShapePartOffsetPrecisions[i][1].ObservedValue = offsetPrecisions[i][1];
                gridHolder.ShapePartPresenseProbability[i].ObservedValue = partPresenseProbs[i];
            }

            // Sample
            bool[][,] result = new bool[count][,];
            for (int i = 0; i < count; ++i)
            {
                result[i] = gridHolder.Sample();
            }

            return result;
        }

        //private static void ReconstructShapeModel(
        //    int shapePartCount,
        //    bool[,] observedLabels,
        //    double observationNoiseProb,
        //    double pottsPenalty,
        //    Tuple<PositiveDefiniteMatrix, Vector>[] trueShapeParams,
        //    out Tuple<PositiveDefiniteMatrix, Vector>[] shapeParams)
        //{
        //    // Specify detection model
        //    //var gridHolder = new ImageModelFullCovariance(GridSize, GridSize, shapePartCount);
        //    var gridHolder = new ImageModelFactorized(GridSize, GridSize, shapePartCount);
        //    gridHolder.ObservedLabels.ObservedValue = observedLabels;
        //    gridHolder.PottsPenalty.ObservedValue = pottsPenalty;
        //    gridHolder.ObservationNoiseProbability.ObservedValue = observationNoiseProb;
        //    for (int i = 0; i < shapePartCount; ++i)
        //    {
        //        gridHolder.ShapeOrientation[i].ObservedValue = trueShapeParams[i].Item1;
        //    }

        //    // Break symmetry by randomly initializing the messages
        //    for (int i = 0; i < shapePartCount; ++i)
        //    {
        //        Vector randomEllipseMean = Vector.FromArray(0.4 + 0.2 * Rand.Double(), 0.4 + 0.2 * Rand.Double());
        //        PositiveDefiniteMatrix randomEllipsePrecision = EllipsePrecisionMatrix(
        //            0.1 + Rand.Double() * 0.1, 0.1 + Rand.Double() * 0.1, Rand.Double() * Math.PI);
        //        PositiveDefiniteMatrix randomEllipseCovariance = new PositiveDefiniteMatrix(2, 2);
        //        randomEllipseCovariance.SetToInverse(randomEllipsePrecision);

        //        const double firstShape = 3.0, secondPrecisionScale = 1.0;
        //        //gridHolder.ShapeParams[i].InitialiseTo<VectorGaussianWishart>(
        //        //    new VectorGaussianWishart(firstShape, randomEllipseCovariance * firstShape, randomEllipseMean, secondPrecisionScale));
        //        //gridHolder.ShapeLocation[i].InitialiseTo<VectorGaussian>(
        //        //    new VectorGaussian(randomEllipseMean, PositiveDefiniteMatrix.IdentityScaledBy(2, 0.2 * 0.2)));
        //    }

        //    InferenceEngine trainingEngine = CreateInferenceEngine();
        //    //var shapeParamsPosterior = Util.ArrayInit(shapePartCount, i => trainingEngine.Infer<VectorGaussianWishart>(gridHolder.ShapeParams[i]));
        //    var shapeXPosterior = Util.ArrayInit(shapePartCount, i => trainingEngine.Infer<Gaussian>(gridHolder.ShapeLocation[i][0]));
        //    var shapeYPosterior = Util.ArrayInit(shapePartCount, i => trainingEngine.Infer<Gaussian>(gridHolder.ShapeLocation[i][1]));
        //    //shapeParams = Util.ArrayInit(shapePartCount, i => shapeParamsPosterior[i].GetMean());
        //    shapeParams = Util.ArrayInit(
        //        shapePartCount,
        //        i => Tuple.Create(trueShapeParams[i].Item1, Vector.FromArray(shapeXPosterior[i].GetMean(), shapeYPosterior[i].GetMean())));
        //}

        private static void LearnShapeModel(
            IList<PositiveDefiniteMatrix> shapeOrientations,
            bool[][,] trainingSet,
            double observationNoiseProb,
            double pottsPenalty)
        {
            int shapePartCount = shapeOrientations.Count;

            var gridHolder = new ImageModelFactorized(GridSize, GridSize, shapePartCount, trainingSet.Length);
            gridHolder.PottsPenalty.ObservedValue = pottsPenalty;
            gridHolder.ObservationNoiseProbability.ObservedValue = observationNoiseProb;
            gridHolder.ObservedLabels.ObservedValue = trainingSet;
            for (int i = 0; i < shapePartCount; ++i)
            {
                gridHolder.ShapePartOrientation[i].ObservedValue = shapeOrientations[i];
            }

            InferenceEngine engine = CreateInferenceEngine();

            for (int iterationCount = 10; iterationCount <= 100; iterationCount += 10)
            {
                engine.NumberOfIterations = iterationCount;

                // Infer part locations
                IList<Gaussian>[] shapePartLocationX = Util.ArrayInit(
                    shapePartCount,
                    i => engine.Infer<IList<Gaussian>>(gridHolder.ShapePartLocation[i][0]));
                IList<Gaussian>[] shapePartLocationY = Util.ArrayInit(
                    shapePartCount,
                    i => engine.Infer<IList<Gaussian>>(gridHolder.ShapePartLocation[i][1]));

                // Draw part locations
                for (int i = 0; i < trainingSet.Length; ++i)
                {
                    var shapeParams = Util.ArrayInit(
                        shapePartCount,
                        j => Tuple.Create(
                            shapeOrientations[j],
                            Vector.FromArray(shapePartLocationX[j][i].GetMean(), shapePartLocationY[j][i].GetMean())));
                    DrawShape(shapeParams).Save(string.Format("fitted_object_{0}_{1}.png", iterationCount, i));
                }

                // Infer shape model
                Gaussian[] shapeOffsetMeanXPosteriors = Util.ArrayInit(
                    shapePartCount,
                    i => engine.Infer<Gaussian>(gridHolder.ShapePartOffsetMeans[i][0]));
                Gaussian[] shapeOffsetMeanYPosteriors = Util.ArrayInit(
                    shapePartCount,
                    i => engine.Infer<Gaussian>(gridHolder.ShapePartOffsetMeans[i][1]));
                Gamma[] shapeOffsetPrecXPosteriors = Util.ArrayInit(
                    shapePartCount,
                    i => engine.Infer<Gamma>(gridHolder.ShapePartOffsetPrecisions[i][0]));
                Gamma[] shapeOffsetPrecYPosteriors = Util.ArrayInit(
                    shapePartCount,
                    i => engine.Infer<Gamma>(gridHolder.ShapePartOffsetPrecisions[i][1]));
                Beta[] shapePartPresesnseProb = Util.ArrayInit(
                    shapePartCount,
                    i => engine.Infer<Beta>(gridHolder.ShapePartPresenseProbability[i]));

                Console.WriteLine("Offset mean X: {0}", StringUtil.ArrayToString(shapeOffsetMeanXPosteriors));
                Console.WriteLine("Offset mean Y: {0}", StringUtil.ArrayToString(shapeOffsetMeanYPosteriors));
                Console.WriteLine("Offset prec X: {0}", StringUtil.ArrayToString(shapeOffsetPrecXPosteriors));
                Console.WriteLine("Offset prec Y: {0}", StringUtil.ArrayToString(shapeOffsetPrecYPosteriors));
                Console.WriteLine("Presense prob: {0}", StringUtil.ArrayToString(shapePartPresesnseProb));
            }
        }

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
            var partPresenseProbs = new[] { 1.0, 1.0, 1.0 };
            var offsetMeans = new[] { Vector.FromArray(0.0, 0.0), Vector.FromArray(0.0, -0.2), Vector.FromArray(0.0, 0.2) };
            var offsetPrecisions = new[] { Vector.FromArray(1.0 / (0.01 * 0.01), 1.0 / (0.01 * 0.01)), Vector.FromArray(1.0 / (0.01 * 0.01), 1.0 / (0.01 * 0.01)), Vector.FromArray(1.0 / (0.1 * 0.1), 1.0 / (0.01 * 0.01)) };
            var shapeOrientations = new[] { EllipsePrecisionMatrix(0.1, 0.2, 0), EllipsePrecisionMatrix(0.02, 0.15, 0), EllipsePrecisionMatrix(0.2, 0.02, 0) };
            var shapeLocationPrior = new[] { Gaussian.FromMeanAndVariance(0.5, 0.1 * 0.1), Gaussian.FromMeanAndVariance(0.5, 0.1 * 0.1) };

            // Sample labels from true model

            bool[][,] sampledLabels = SampleNoisyLabels(
                shapeOrientations, offsetMeans, offsetPrecisions, partPresenseProbs, shapeLocationPrior, observationNoiseProb, pottsPenalty, trainingSetSize);

            // Save samples
            for (int i = 0; i < sampledLabels.Length; ++i)
            {
                ImageHelpers.ArrayToBitmap(sampledLabels[i], b => b ? Color.Red : Color.Green).Save(string.Format("sampled_labels_{0}.png", i));
            }

            // Add redundant shapes
            var shapeOrientationsWithRedundant = new List<PositiveDefiniteMatrix>(shapeOrientations);
            const int RedundantShapeCount = 1;
            for (int i = 0; i < RedundantShapeCount; ++i)
            {
                double randomWidth = 0.02 + 0.2 * Rand.Double();
                double randomHeight = 0.02 + 0.2 * Rand.Double();
                double randomAngle = Math.PI * 2 * Rand.Double();
                shapeOrientationsWithRedundant.Add(EllipsePrecisionMatrix(randomWidth, randomHeight, randomAngle));
            }

            LearnShapeModel(shapeOrientationsWithRedundant, sampledLabels, observationNoiseProb, pottsPenalty);
        }

        //private static void MainSampleRestore()
        //{
        //    Rand.Restart(666);

        //    const double observationNoiseProb = 0.01;
        //    const double pottsPenalty = 0.9;

        //    // 1 ellipse
        //    var shapeParams = new[]
        //    {
        //        Tuple.Create(EllipsePrecisionMatrix(0.3, 0.1, Math.PI * 0.25), Vector.FromArray(0.4, 0.5)),
        //    };

        //    // 2 ellipses
        //    //var shapeParams = new[]
        //    //{
        //    //    Tuple.Create(EllipsePrecisionMatrix(0.45, 0.08, Math.PI * 0.25), Vector.FromArray(0.4, 0.5)),
        //    //    Tuple.Create(EllipsePrecisionMatrix(0.45, 0.08, -Math.PI * 0.25), Vector.FromArray(0.5, 0.5))
        //    //};

        //    // 3 ellipses
        //    //var shapeParams = new[]
        //    //{
        //    //    Tuple.Create(EllipsePrecisionMatrix(0.06, 0.45, 0.0), Vector.FromArray(0.5, 0.5)),
        //    //    Tuple.Create(EllipsePrecisionMatrix(0.3, 0.06, 0.0), Vector.FromArray(0.5, 0.3)),
        //    //    Tuple.Create(EllipsePrecisionMatrix(0.17, 0.05, Math.PI * 0.15), Vector.FromArray(0.5, 0.7))
        //    //};

        //    // Draw true model
        //    DrawShape(shapeParams).Save("true_object_vs_bg.png");

        //    // Sample labels from true model
        //    bool[,] sampledLabels, sampledObservedLabels;
        //    SampleLabels(shapeParams, observationNoiseProb, pottsPenalty, out sampledLabels, out sampledObservedLabels);

        //    // Save samples
        //    ImageHelpers.ArrayToBitmap(sampledLabels, b => b ? Color.Red : Color.Green).Save("sampled_labels.png");
        //    ImageHelpers.ArrayToBitmap(sampledObservedLabels, b => b ? Color.Red : Color.Green).Save("sampled_observed_labels.png");

        //    // Reconstruct shape
        //    Tuple<PositiveDefiniteMatrix, Vector>[] learnedShapeParams;
        //    ReconstructShapeModel(
        //        shapeParams.Length,
        //        sampledObservedLabels,
        //        observationNoiseProb,
        //        pottsPenalty,
        //        shapeParams,
        //        out learnedShapeParams);

        //    // Output detection results
        //    DrawShape(learnedShapeParams).Save("learned_object_vs_bg.png");

        //    // Sample from reconstructed model
        //    //bool[,] sampledLabels2, sampledObservedLabels2;
        //    //SampleLabels(learnedShapeParamsX, learnedShapeParamsY, observationNoiseProb, pottsPenalty, out sampledLabels2, out sampledObservedLabels2);

        //    //// Save samples
        //    //ImageHelpers.ArrayToBitmap(sampledLabels2, b => b ? Color.Red : Color.Green).Save("sampled_labels_2.png");
        //    //ImageHelpers.ArrayToBitmap(sampledObservedLabels2, b => b ? Color.Red : Color.Green).Save("sampled_observed_labels_2.png");
        //}

        private static PositiveDefiniteMatrix EllipsePrecisionMatrix(
            double stdDev1, double stdDev2, double angle)
        {
            PositiveDefiniteMatrix covariance = new PositiveDefiniteMatrix(2, 2);
            covariance[0, 0] = stdDev1 * stdDev1;
            covariance[1, 1] = stdDev2 * stdDev2;

            PositiveDefiniteMatrix rotationMatrix = RotationMatrix(angle);
            PositiveDefiniteMatrix rotationMatrixInverse = RotationMatrix(-angle);

            PositiveDefiniteMatrix rotatedCovariance = new PositiveDefiniteMatrix(2, 2);
            rotatedCovariance.SetTo(rotationMatrix * covariance * rotationMatrixInverse);

            return rotatedCovariance.Inverse();
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
