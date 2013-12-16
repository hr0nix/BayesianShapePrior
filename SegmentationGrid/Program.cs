using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Drawing.Imaging;

using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Utils;

using ShapePart = System.Tuple<MicrosoftResearch.Infer.Maths.PositiveDefiniteMatrix, MicrosoftResearch.Infer.Maths.Vector>;

namespace SegmentationGrid
{
    using BernoulliArray1D = DistributionStructArray<Bernoulli, bool>;
    using BernoulliArray2D = DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>;
    using GaussianArray1D = DistributionStructArray<Gaussian, double>;
    using GaussianArray2D = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using GaussianArray3D = DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>, double[][]>;
    using GammaArray1D = DistributionStructArray<Gamma, double>;
    using GammaArray2D = DistributionRefArray<DistributionStructArray<Gamma, double>, double[]>;
    using WishartArray1D = DistributionRefArray<Wishart, PositiveDefiniteMatrix>;
    using WishartArray2D = DistributionRefArray<DistributionRefArray<Wishart, PositiveDefiniteMatrix>, PositiveDefiniteMatrix[]>;
    using ShapePartArray2D = DistributionRefArray<DistributionRefArray<VectorGaussianWishart, ShapePart>, ShapePart[]>;
    using LabelArray = DistributionRefArray<DistributionStructArray2D<Bernoulli, bool>, bool[,]>;
    
    class Program
    {
        private static void Main()
        {
            Rand.Restart(666);

            const double observationNoiseProb = 0.2;
            const double pottsPenalty = 0.9;

            // 1 ellipse
            //var offsetMeans = new[] { Vector.FromArray(0.0, 0.0) };
            //var offsetPrecisions = new[] { Vector.FromArray(1.0 / (0.1 * 0.1), 1.0 / (0.1 * 0.1)) };
            //var orientationNoiseShape = 10;
            //var shapeOrientationRates = new[] { EllipsePrecisionMatrix(0.22, 0.08, Math.PI * 0.33).Inverse() * orientationNoiseShape };
            //var shapeLocationPrior = new[] { Gaussian.FromMeanAndVariance(0.5, 0.1 * 0.1), Gaussian.FromMeanAndVariance(0.5, 0.1 * 0.1) };

            // 2 ellipses 1
            //var offsetMeans = new[] { Vector.FromArray(0.0, 0.0), Vector.FromArray(0.0, 0.0) };
            //var offsetPrecisions = new[] { Vector.FromArray(1.0 / (0.05 * 0.05), 1.0 / (0.05 * 0.05)), Vector.FromArray(1.0 / (0.05 * 0.05), 1.0 / (0.05 * 0.05)) };
            //var orientationNoiseShape = 10;
            //var shapeOrientationRates = new[]
            //{
            //    EllipsePrecisionMatrix(0.3, 0.05, Math.PI * 0.25).Inverse() * orientationNoiseShape,
            //    EllipsePrecisionMatrix(0.2, 0.07, Math.PI * 0.75).Inverse() * orientationNoiseShape
            //};
            //var shapeLocationPrior = new[] { Gaussian.FromMeanAndVariance(0.5, 0.2 * 0.2), Gaussian.FromMeanAndVariance(0.5, 0.2 * 0.2) };

            // 2 ellipses 2
            //var offsetMeans = new[] { Vector.FromArray(0.0, -0.17), Vector.FromArray(0.0, 0.0) };
            //var offsetPrecisions = new[] { Vector.FromArray(1.0 / (0.05 * 0.05), 1.0 / (0.05 * 0.05)), Vector.FromArray(1.0 / (0.05 * 0.05), 1.0 / (0.05 * 0.05)) };
            //var orientationNoiseShape = 10;
            //var shapeOrientationRates = new[]
            //{
            //    EllipsePrecisionMatrix(0.2, 0.05, 0.0).Inverse() * orientationNoiseShape,
            //    EllipsePrecisionMatrix(0.05, 0.3, 0.0).Inverse() * orientationNoiseShape
            //};
            //var shapeLocationPrior = new[] { Gaussian.FromMeanAndVariance(0.5, 0.2 * 0.2), Gaussian.FromMeanAndVariance(0.5, 0.2 * 0.2) };

            // 3 ellipses 1
            //var offsetMeans = new[] { Vector.FromArray(0.0, 0.0), Vector.FromArray(0.0, -0.2), Vector.FromArray(0.0, 0.2) };
            //var offsetPrecisions = new[] { Vector.FromArray(1.0 / (0.01 * 0.01), 1.0 / (0.01 * 0.01)), Vector.FromArray(1.0 / (0.01 * 0.01), 1.0 / (0.01 * 0.01)), Vector.FromArray(1.0 / (0.1 * 0.1), 1.0 / (0.01 * 0.01)) };
            //var orientationNoiseShape = 10;
            //var shapeOrientationRates = new[]
            //{
            //    EllipsePrecisionMatrix(0.1, 0.2, 0).Inverse() * orientationNoiseShape,
            //    EllipsePrecisionMatrix(0.02, 0.15, 0).Inverse() * orientationNoiseShape,
            //    EllipsePrecisionMatrix(0.2, 0.02, 0).Inverse() * orientationNoiseShape
            //};
            //var shapeLocationPrior = new[] { Gaussian.FromMeanAndVariance(0.5, 0.1 * 0.1), Gaussian.FromMeanAndVariance(0.5, 0.1 * 0.1) };

            // 3 ellipses 2
            //var offsetMeans = new[] { Vector.FromArray(0.0, 0.0), Vector.FromArray(-0.2, 0.3), Vector.FromArray(0.2, 0.3) };
            //var offsetPrecisions = new[] { Vector.FromArray(1.0 / (0.02 * 0.02), 1.0 / (0.02 * 0.02)), Vector.FromArray(1.0 / (0.02 * 0.02), 1.0 / (0.02 * 0.02)), Vector.FromArray(1.0 / (0.02 * 0.02), 1.0 / (0.02 * 0.02)) };
            //var orientationNoiseShape = 10;
            //var shapeOrientationRates = new[]
            //{
            //    EllipsePrecisionMatrix(0.3, 0.1, 0).Inverse() * orientationNoiseShape,
            //    EllipsePrecisionMatrix(0.03, 0.3, 0).Inverse() * orientationNoiseShape,
            //    EllipsePrecisionMatrix(0.03, 0.3, 0).Inverse() * orientationNoiseShape
            //};
            //var shapeLocationPrior = new[] { Gaussian.FromMeanAndVariance(0.5, 0.1 * 0.1), Gaussian.FromMeanAndVariance(0.5, 0.1 * 0.1) };

            // Sample labels from true model

            //bool[][,] noisyLabels;
            //double[][][] locations;
            //PositiveDefiniteMatrix[][] orientations;
            //const int trainingSetSize = 10;
            //const int gridSize = 100;
            //SampleNoisyLabels(
            //    offsetMeans,
            //    offsetPrecisions,
            //    shapeLocationPrior,
            //    shapeOrientationRates,
            //    orientationNoiseShape,
            //    observationNoiseProb,
            //    pottsPenalty,
            //    trainingSetSize,
            //    gridSize,
            //    out noisyLabels,
            //    out locations,
            //    out orientations);

            //// Draw the desired shape model
            //DrawShapeModel(offsetMeans, shapeOrientationRates, orientationNoiseShape).Save("true_shape_model.png");

            //int shapePartCount = shapeOrientationRates.Length;

            bool[][,] labels = LoadLabels("../../../Data/horses/figure_ground", 30);
            bool[][,] noisyLabels = labels;// Util.ArrayInit(labels.Length, i => MakeNoisyMask(labels[i], observationNoiseProb));
            const int shapePartCount = 8;
            const int traitCount = 2;

            // Save training set
            for (int i = 0; i < noisyLabels.Length; ++i)
            {
                ImageHelpers.ArrayToBitmap(noisyLabels[i], b => b ? Color.Red : Color.Green).Save(string.Format("noisy_labels_{0}.png", i));
            }

            var shapeModel = LearnShapeModel(
                noisyLabels[0].GetLength(0), noisyLabels[0].GetLength(1), shapePartCount, traitCount, noisyLabels, observationNoiseProb, pottsPenalty);

            bool[][,] testLabels = LoadLabels("../../../Data/horses/figure_ground", 30, labels.Length);
            
            // Save test set
            for (int i = 0; i < testLabels.Length; ++i)
            {
                ImageHelpers.ArrayToBitmap(testLabels[i], b => b ? Color.Red : Color.Green).Save(string.Format("test_labels_{0}.png", i));
            }

            CompleteImages(
                testLabels[0].GetLength(0), testLabels[0].GetLength(1), testLabels, observationNoiseProb, pottsPenalty, shapeModel);
        }

        private static bool[][,] LoadLabels(string path, int maxImagesToLoad, int skip = 0)
        {
            List<Bitmap> bitmaps = new List<Bitmap>();

            int minWidth = int.MaxValue, minHeight = int.MaxValue;
            foreach (string imagePath in Directory.EnumerateFiles(path))
            {
                if (skip-- > 0)
                {
                    continue;
                }
                
                if (bitmaps.Count == maxImagesToLoad)
                {
                    break;
                }

                Bitmap bitmap = new Bitmap(imagePath);
                bitmaps.Add(bitmap);
                minWidth = Math.Min(minWidth, bitmap.Width);
                minHeight = Math.Min(minHeight, bitmap.Height);
            }

            bool[][,] result = new bool[bitmaps.Count][,];
            for (int i = 0; i < bitmaps.Count; ++i)
            {
                Rectangle subImageRect = new Rectangle((bitmaps[i].Width - minWidth) / 2, (bitmaps[i].Height - minHeight) / 2, minWidth, minHeight);
                Bitmap croppedBitmap = bitmaps[i].Clone(subImageRect, PixelFormat.Format24bppRgb);
                result[i] = ImageHelpers.BitmapToArray(croppedBitmap, c => c.R > 0);
            }

            return result;
        }

        private static void CompleteImages(
            int gridWidth,
            int gridHeight,
            bool[][,] images,
            double observationNoiseProb,
            double pottsPenalty,
            ImageModelFactorized.ShapeModelBelief shapeModel)
        {
            var gridHolder = new ImageModelFactorized(
                gridWidth, gridHeight, shapeModel.ShapePartOffsetWeights.Count, shapeModel.ShapePartOffsetWeights[0][0].Count, images.Length);
            gridHolder.OptimizeForLabelCompletion();
            gridHolder.PottsPenalty.ObservedValue = pottsPenalty;
            gridHolder.ObservationNoiseProbability.ObservedValue = observationNoiseProb;

            gridHolder.SetShapeModel(shapeModel);

            Bernoulli[][,] constraints = new Bernoulli[images.Length][,];
            for (int imageIndex = 0; imageIndex < images.Length; ++imageIndex)
            {
                constraints[imageIndex] = new Bernoulli[images[imageIndex].GetLength(0), images[imageIndex].GetLength(1)];
                for (int i = 0; i < images[imageIndex].GetLength(0); ++i)
                {
                    for (int j = 0; j < images[imageIndex].GetLength(1); ++j)
                    {
                        if (i < images[imageIndex].GetLength(0) / 2)
                        {
                            constraints[imageIndex][i, j] = Bernoulli.Uniform();
                        }
                        else
                        {
                            constraints[imageIndex][i, j] = Bernoulli.PointMass(images[imageIndex][i, j]);
                        }
                    }
                }
            }

            gridHolder.NoisyLabelsConstraint.ObservedValue = constraints;

            gridHolder.Engine.NumberOfIterations = 1000;
            var labelsPosterior = gridHolder.Engine.Infer <LabelArray>(gridHolder.Labels);
            for (int imageIndex = 0; imageIndex < labelsPosterior.Count; ++imageIndex)
            {
                Bernoulli[,] labels = Util.ArrayInit(
                    labelsPosterior[imageIndex].GetLength(0),
                    labelsPosterior[imageIndex].GetLength(1),
                    (i, j) => i < images[imageIndex].GetLength(0) / 2 ? labelsPosterior[imageIndex][i, j] : Bernoulli.PointMass(images[imageIndex][i, j]));
                ImageHelpers.ArrayToBitmap(labels, b => b.IsPointMass ? (b.Point ? Color.Yellow : Color.Green) : PenaltyToColor(b.GetProbTrue(), 0, 1)).Save("./restored_image_" + imageIndex + ".png");
            }
        }

        private static ImageModelFactorized.ShapeModelBelief LearnShapeModel(
            int gridWidth,
            int gridHeight,
            int shapePartCount,
            int traitCount,
            bool[][,] trainingSet,
            double observationNoiseProb,
            double pottsPenalty)
        {            
            int observationCount = trainingSet.Length;

            var gridHolder = new ImageModelFactorized(gridWidth, gridHeight, shapePartCount, traitCount, observationCount);
            gridHolder.OptimizeForShapeModelLearning();
            gridHolder.PottsPenalty.ObservedValue = pottsPenalty;
            gridHolder.ObservationNoiseProbability.ObservedValue = observationNoiseProb;
            gridHolder.NoisyLabels.ObservedValue = trainingSet;

            var samplingGridHolder = new ImageModelFactorized(gridWidth, gridHeight, shapePartCount, traitCount, observationCount);
            samplingGridHolder.PottsPenalty.ObservedValue = pottsPenalty;
            samplingGridHolder.ObservationNoiseProbability.ObservedValue = observationNoiseProb;

            for (int iterationCount = 100; iterationCount <= 1000; iterationCount += 100)
            {
                gridHolder.Engine.NumberOfIterations = iterationCount;

                //GaussianArray2D shapeTraits = gridHolder.Engine.Infer<GaussianArray2D>(gridHolder.ShapeTraits);
                //for (int i = 0; i < shapeTraits.Count; ++i)
                //{
                //    Console.WriteLine("Image {0} traits:  {1}", i, Util.CollectionToString(shapeTraits[i]));
                //}
                
                GaussianArray3D shapePartLocations = gridHolder.Engine.Infer<GaussianArray3D>(gridHolder.ShapePartLocation);
                WishartArray2D shapePartOrientations = gridHolder.Engine.Infer<WishartArray2D>(gridHolder.ShapePartOrientation);
                for (int i = 0; i < trainingSet.Length; ++i)
                {
                    Bitmap drawnShape = DrawShape(
                        gridWidth,
                        gridHeight,
                        Util.ArrayInit(
                            shapePartCount, j => Tuple.Create(shapePartOrientations[i][j].GetMean(), Vector.FromArray(shapePartLocations[i][j][0].GetMean(), shapePartLocations[i][j][1].GetMean()))));
                    drawnShape.Save(string.Format("fitted_object_{0}_{1}.png", iterationCount, i));
                }

                // Infer shape model
                ImageModelFactorized.ShapeModelBelief shapeModel = gridHolder.InferShapeModel();

                // Draw shape model
                //DrawShapeModel(gridWidth, gridHeight, shapeModel.ShapePartOffsetMeans, shapeModel.ShapePartOrientationRates, gridHolder.ShapePartOrientationShape.ObservedValue).Save(string.Format("shape_model_{0}.png", iterationCount));
                
                // Draw samples from shape model
                samplingGridHolder.SetShapeModel(shapeModel);
                
                bool[][,] sampledLabelsCentered = samplingGridHolder.Sample(10, false, false, true);
                for (int sampleIndex = 0; sampleIndex < sampledLabelsCentered.Length; ++sampleIndex)
                {
                    string sampleFileName = string.Format("labels_centered_sample_{0:000}_{1:00}.png", iterationCount, sampleIndex);
                    ImageHelpers.ArrayToBitmap(sampledLabelsCentered[sampleIndex], l => l ? Color.Red : Color.Green).Save(sampleFileName);
                }

                bool[][,] sampledLabels = samplingGridHolder.Sample(10, false, false, false);
                for (int sampleIndex = 0; sampleIndex < sampledLabelsCentered.Length; ++sampleIndex)
                {
                    string sampleFileName = string.Format("labels_sample_{0:000}_{1:00}.png", iterationCount, sampleIndex);
                    ImageHelpers.ArrayToBitmap(sampledLabels[sampleIndex], l => l ? Color.Red : Color.Green).Save(sampleFileName);
                }
            }

            return gridHolder.InferShapeModel();
        }

        private static bool[,] MakeNoisyMask(bool[,] mask, double flipProb)
        {
            return Util.ArrayInit(mask.GetLength(0), mask.GetLength(1), (i, j) => Rand.Double() < flipProb ? !mask[i, j] : mask[i, j]);
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

        private static Bitmap DrawShapeModel(int gridWidth, int gridHeight, GaussianArray2D shapePartLocationOffsetMeans, WishartArray1D shapePartOrientationRates, double shapePartOrientationShape)
        {
            return DrawShapeModel(
                gridWidth,
                gridHeight,
                Util.ArrayInit(shapePartLocationOffsetMeans.Count, i => Vector.FromArray(shapePartLocationOffsetMeans[i][0].GetMean(), shapePartLocationOffsetMeans[i][1].GetMean())),
                Util.ArrayInit(shapePartOrientationRates.Count, i => shapePartOrientationRates[i].GetMean()),
                shapePartOrientationShape);
        }

        private static Bitmap DrawShapeModel(int gridWidth, int gridHeight, double[][] shapePartLocationOffsetMeans, PositiveDefiniteMatrix[] shapePartOrientationRates, double shapePartOrientationShape)
        {
            return DrawShapeModel(
                gridWidth,
                gridHeight,
                Util.ArrayInit(shapePartLocationOffsetMeans.Length, i => Vector.FromArray(shapePartLocationOffsetMeans[i][0], shapePartLocationOffsetMeans[i][1])),
                shapePartOrientationRates,
                shapePartOrientationShape);
        }
        
        private static Bitmap DrawShapeModel(int gridWidth, int gridHeight, Vector[] shapePartLocationOffsetMeans, PositiveDefiniteMatrix[] shapePartOrientationRates, double shapePartOrientationShape)
        {
            Bitmap result = new Bitmap(gridWidth, gridHeight);
            using (Graphics graphics = Graphics.FromImage(result))
            {
                graphics.Clear(Color.Black);
                
                Vector centerPos = Vector.FromArray(0.5, 0.5);    
                for (int i = 0; i < shapePartLocationOffsetMeans.Length; ++i)
                {
                    Vector shapePartPos = centerPos + shapePartLocationOffsetMeans[i];
                    graphics.DrawLine(Pens.Green, gridWidth * (float)centerPos[0], gridHeight * (float)centerPos[1], gridWidth * (float)shapePartPos[0], gridHeight * (float)shapePartPos[1]);
                    DrawShapeEllipse(gridWidth, gridHeight, graphics, Pens.Red, shapePartPos, shapePartOrientationRates[i].Inverse() * shapePartOrientationShape);
                }
            }

            return result;
        }

        private static void DrawShapeEllipse(int gridWidth, int gridHeight, Graphics graphics, Pen pen, Vector pos, PositiveDefiniteMatrix ellipsePrecisionMatrix)
        {
            PositiveDefiniteMatrix inverse = ellipsePrecisionMatrix.Inverse();
            LowerTriangularMatrix cholesky = new LowerTriangularMatrix(2, 2);
            cholesky.SetToCholesky(inverse);

            const int points = 20;
            Vector prevPoint = null;
            double width = 2 * Math.Log(2);
            for (int i = 0; i <= points; ++i)
            {
                double angle = Math.PI * 2 * i / points;
                Vector point = pos + cholesky * Vector.FromArray(Math.Cos(angle), Math.Sin(angle)) * width;
                if (prevPoint != null)
                {
                    graphics.DrawLine(pen, gridWidth * (float) prevPoint[0], gridHeight * (float) prevPoint[1], gridWidth * (float) point[0], gridHeight * (float) point[1]);
                }

                prevPoint = point;
            }
        }
        
        private static Bitmap DrawShape(int gridWidth, int gridHeight, IEnumerable<ShapePart> shapeParts)
        {
            double[,] mask = Util.ArrayInit(
                gridWidth,
                gridHeight,
                (x, y) =>
                {
                    double prob = 1.0;
                    foreach (ShapePart part in shapeParts)
                    {
                        Bernoulli partLabelDistr = LabelFromShapeFullCovarianceOps.LabelAverageConditional(
                            Vector.FromArray((x + 0.5) / gridWidth, (y + 0.5) / gridHeight), part);
                        prob *= (1 - partLabelDistr.GetProbTrue());
                    }

                    return 1 - prob;
                });

            Bitmap result = ImageHelpers.ArrayToBitmap(mask, p => PenaltyToColor(p, 0, 1));

            using (Graphics graphics = Graphics.FromImage(result))
            {
                int index = 1;
                foreach (ShapePart shapePart in shapeParts)
                {
                    Color color = Color.FromArgb(255 * (index & 1), 255 * ((index >> 1) & 1), 255 * ((index >> 2) & 1));
                    Pen pen = new Pen(color);
                    DrawShapeEllipse(gridWidth, gridHeight, graphics, pen, shapePart.Item2, shapePart.Item1);
                    ++index;
                }
            }

            return result;
        }

        private static Bitmap DrawShape(int gridWidth, int gridHeight, IEnumerable<VectorGaussianWishart> shapePartDists)
        {
            double[,] mask = Util.ArrayInit(
                gridWidth,
                gridHeight,
                (x, y) =>
                {
                    double prob = 1.0;
                    foreach (VectorGaussianWishart part in shapePartDists)
                    {
                        Bernoulli partLabelDistr = LabelFromShapeFullCovarianceOps.LabelAverageConditional(
                            Vector.FromArray((x + 0.5) / gridWidth, (y + 0.5) / gridHeight), part);
                        prob *= (1 - partLabelDistr.GetProbTrue());
                    }

                    return 1 - prob;
                });

            Bitmap result = ImageHelpers.ArrayToBitmap(mask, p => PenaltyToColor(p, 0, 1));
            return result;
        }

        private static Bitmap DrawShape(int gridWidth, int gridHeight, VectorGaussianWishart shapePartDist)
        {
            return DrawShape(gridWidth, gridHeight, new[] { shapePartDist });
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
