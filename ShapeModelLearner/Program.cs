using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;

using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Utils;

namespace SegmentationGrid
{
    using LabelArray = DistributionRefArray<DistributionStructArray2D<Bernoulli, bool>, bool[,]>;

    class Program
    {
        private static void AngleScaleTest()
        {
            Variable<double> angle = Variable.GaussianFromMeanAndVariance(Math.PI / 6, 0.4 * 0.4);
            Variable<double> scaleX = Variable.GaussianFromMeanAndVariance(1.5, 2 * 2);
            Variable<double> scaleY = Variable.GaussianFromMeanAndVariance(1.5, 2 * 2);
            Variable<PositiveDefiniteMatrix> prec = Variable<PositiveDefiniteMatrix>.Factor(ShapeFactors.MatrixFromAngleScale, scaleX, scaleY, angle);

            Vector trueMean = Vector.Zero(2);

            Range pointRange = new Range(5000);
            VariableArray<Vector> points = Variable.Array<Vector>(pointRange);
            points[pointRange] = Variable.VectorGaussianFromMeanAndPrecision(trueMean, prec).ForEach(pointRange);

            PositiveDefiniteMatrix truePrec = ShapeFactors.MatrixFromAngleScale(2.0, 3.0, Math.PI / 5);
            Vector[] observedPoints = Util.ArrayInit(pointRange.SizeAsInt, i => VectorGaussian.Sample(trueMean, truePrec));

            points.ObservedValue = observedPoints;
            prec.AddAttribute(new PointEstimate());
            prec.AddAttribute(new MarginalPrototype(new Wishart(2)));

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.RequiredQuality = engine.Compiler.RecommendedQuality = QualityBand.Unknown;
            Console.WriteLine(engine.Infer(angle));
            Console.WriteLine(engine.Infer(scaleX));
            Console.WriteLine(engine.Infer(scaleY));
        }

        private static void Main()
        {
            Rand.Restart(666);

            //AngleScaleTest();
            //return;

            const int shapePartCount = 8;
            const int traitCount = 5;
            const double observationNoiseProb = 0.01;
            const int trainingSetSize = 30;
            const int imageRepeatCount = 1;
            const string modelNamePrefx = "horses";
            //const string trainTestImagePath = "../../../Data/Caltech101/AnnotationsConverted/Motorbikes_16";
            //const string trainTestImagePath = "../../../Data/horses/scale_invariance";
            const string trainTestImagePath = "../../../Data/horses/clean";

            bool[][,] trainingLabels = ImageHelpers.LoadImagesAsSameSizeMasks(trainTestImagePath, trainingSetSize, trainingSetSize, 0);
            trainingLabels = RepeatImages(trainingLabels, imageRepeatCount);

            // Save training set
            for (int i = 0; i < trainingLabels.Length; ++i)
            {
                ImageHelpers.ArrayToBitmap(trainingLabels[i], b => b ? Color.Red : Color.Green).Save(string.Format("training_labels_{0}.png", i));
            }

            ShapeModelInference inference = new ShapeModelInference();
            inference.TrainingIterationCount = 10000;
            inference.IterationsBetweenCallbacks = 1;

            inference.SetPriorBeliefs(trainingLabels[0].GetLength(0), trainingLabels[0].GetLength(1), traitCount, shapePartCount, observationNoiseProb);
            inference.ShapeModelLearningProgress += OnShapeModelLearningProgress;
            ShapeModel model = inference.LearnModel(trainingLabels).Item1;

            model.Save(string.Format("./{0}_{1}_traits_{2}_images_{3}_parts.bin", modelNamePrefx, traitCount, trainingSetSize, shapePartCount));
        }

        static bool[][,] RepeatImages(bool[][,] traininglabels, int repeatCount)
        {
            bool[][,] result = new bool[traininglabels.Length * repeatCount][,];
            for (int i = 0; i < result.Length; ++i)
            {
                result[i] = traininglabels[i / repeatCount];
            }

            return result;
        }

        static void OnShapeModelLearningProgress(object sender, ShapeModelLearningProgressEventArgs e)
        {
            e.ShapeModel.Save(string.Format("model_progress_{0}.bin", e.IterationsCompleted));

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("After iteration {0}:", e.IterationsCompleted);
            Console.WriteLine();
            Console.ResetColor();

            for (int i = 0; i < e.FittingInfo.ShapeTraits.Count; ++i)
            {
                Console.Write("Trait means {0}:", i);
                for (int j = 0; j < e.FittingInfo.ShapeTraits[i].Count; ++j)
                {
                    Console.Write("  {0:0.000} (var={1:0.000})", e.FittingInfo.ShapeTraits[i][j].GetMean(), e.FittingInfo.ShapeTraits[i][j].GetVariance());
                }
                Console.WriteLine();
            }
            Console.WriteLine();

            Console.WriteLine("Scale trait weights:");
            for (int i = 0; i < e.ShapeModel.ShapePartCount; ++i)
            {
                Console.WriteLine("Shape part {0}", i);

                Console.Write("X:");
                for (int j = 0; j < e.ShapeModel.TraitCount; ++j)
                {
                    Console.Write("  {0:0.0000} (std={1:0.0000})", e.ShapeModel.ShapePartLogScaleWeights[i][0][j].GetMean(), Math.Sqrt(e.ShapeModel.ShapePartLogScaleWeights[i][0][j].GetVariance()));
                }
                Console.WriteLine();

                Console.Write("Y:");
                for (int j = 0; j < e.ShapeModel.TraitCount; ++j)
                {
                    Console.Write("  {0:0.0000} (std={1:0.0000})", e.ShapeModel.ShapePartLogScaleWeights[i][1][j].GetMean(), Math.Sqrt(e.ShapeModel.ShapePartLogScaleWeights[i][1][j].GetVariance()));
                }
                Console.WriteLine();
            }
            Console.WriteLine();

            Console.WriteLine("Angle trait weights:");
            for (int i = 0; i < e.ShapeModel.ShapePartCount; ++i)
            {
                Console.Write("Shape part {0}:", i);
                for (int j = 0; j < e.ShapeModel.TraitCount; ++j)
                {
                    Console.Write("  {0:0.0000} (std={1:0.0000})", e.ShapeModel.ShapePartAngleWeights[i][j].GetMean(), Math.Sqrt(e.ShapeModel.ShapePartAngleWeights[i][j].GetVariance()));
                }
                Console.WriteLine();
            }
            Console.WriteLine();

            Console.WriteLine("Global log-scales: ");
            for (int i = 0; i < e.FittingInfo.GlobalLogScales.Count; ++i)
            {
                Console.WriteLine("Image {0}: {1:0.0000} (std={2:0.0000})", i, e.FittingInfo.GlobalLogScales[i].GetMean(), Math.Sqrt(e.FittingInfo.GlobalLogScales[i].GetVariance()));
            }
            Console.WriteLine();

            for (int imageIndex = 0; imageIndex < e.FittingInfo.ShapePartLocations.Count; ++imageIndex)
            {
                string fileName = string.Format("shape_{0:000}_{1:00}.png", e.IterationsCompleted, imageIndex);
                Vector[] meanLocations = Util.ArrayInit(e.FittingInfo.ShapePartLocations[imageIndex].Count, i => Vector.FromArray(Util.ArrayInit(2, j => e.FittingInfo.ShapePartLocations[imageIndex][i][j].GetMean())));
                PositiveDefiniteMatrix[] meanOrientations = Util.ArrayInit(e.FittingInfo.ShapePartOrientations[imageIndex].Count, j => e.FittingInfo.ShapePartOrientations[imageIndex][j].GetMean());
                ImageHelpers.DrawShape(e.ShapeModel.GridWidth, e.ShapeModel.GridHeight, meanLocations, meanOrientations).Save(fileName);

                string sampleFileName = string.Format("shape_{0:000}_{1:00}_sample.png", e.IterationsCompleted, imageIndex);
                double[] meanTraits = Util.ArrayInit(e.ShapeModel.TraitCount, i => e.FittingInfo.ShapeTraits[imageIndex][i].GetMean());
                ShapeModelSample sample = e.ShapeModel.Sample(new[] { meanTraits }, false, true)[0];
                ImageHelpers.DrawShape(e.ShapeModel.GridWidth, e.ShapeModel.GridHeight, sample.ShapePartLocations, sample.ShapePartOrientations).Save(sampleFileName);
            }

            //ShapeModelSample[] samples = e.ShapeModel.Sample(10, false, false, true);
            //for (int sampleIndex = 0; sampleIndex < samples.Length; ++sampleIndex)
            //{
            //    string sampleFileName = string.Format("labels_centered_sample_{0:000}_{1:00}.png", e.IterationsCompleted, sampleIndex);
            //    ImageHelpers.ArrayToBitmap(samples[sampleIndex].Labels, l => l ? Color.Red : Color.Green).Save(sampleFileName);
            //}
        }
    }
}
