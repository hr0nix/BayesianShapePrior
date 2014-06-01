using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace SegmentationGrid
{
    using GaussianArray1D = DistributionStructArray<Gaussian, double>;
    using GaussianArray2D = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using GaussianArray3D = DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>, double[][]>;
    using GammaArray1D = DistributionStructArray<Gamma, double>;
    using GammaArray2D = DistributionRefArray<DistributionStructArray<Gamma, double>, double[]>;
    using WishartArray1D = DistributionRefArray<Wishart, PositiveDefiniteMatrix>;
    using WishartArray2D = DistributionRefArray<DistributionRefArray<Wishart, PositiveDefiniteMatrix>, PositiveDefiniteMatrix[]>;
    using LabelArray = DistributionRefArray<DistributionStructArray2D<Bernoulli, bool>, bool[,]>;

    public class ShapeModelSample
    {
        public ShapeModel ShapeModel { get; set; }
        
        public bool[,] Labels { get; set; }

        public Vector[] ShapePartLocations { get; set; }

        public PositiveDefiniteMatrix[] ShapePartOrientations { get; set; }
    }

    [Serializable]
    public class ShapeModel
    {
        public ShapeModel(
            int gridWidth,
            int gridHeight,
            double observationNoiseProb,
            GaussianArray1D shapeLocationMean,
            GammaArray1D shapeLocationPrecision,
            GaussianArray3D shapePartOffsetWeights,
            GaussianArray3D shapePartLogScaleWeights,
            GaussianArray2D shapePartAngleWeights,
            GammaArray2D shapePartOffsetPrecisions,
            GammaArray2D shapePartLogScalePrecisions,
            GammaArray1D shapePartAnglePrecisions)
        {
            this.GridWidth = gridWidth;
            this.GridHeight = gridHeight;
            this.ObservationNoiseProb = observationNoiseProb;
            this.ShapeLocationMean = shapeLocationMean;
            this.ShapeLocationPrecision = shapeLocationPrecision;
            this.ShapePartOffsetWeights = shapePartOffsetWeights;
            this.ShapePartLogScaleWeights = shapePartLogScaleWeights;
            this.ShapePartAngleWeights = shapePartAngleWeights;
            this.ShapePartOffsetPrecisions = shapePartOffsetPrecisions;
            this.ShapePartLogScalePrecisions = shapePartLogScalePrecisions;
            this.ShapePartAnglePrecisions = shapePartAnglePrecisions;
        }

        public int GridWidth { get; private set; }

        public int GridHeight { get; private set; }

        public int ShapePartCount { get { return this.ShapePartOffsetWeights.Count; } }

        public int TraitCount { get { return this.ShapePartOffsetWeights[0][0].Count; } }

        public double ObservationNoiseProb { get; private set; }

        public GaussianArray1D ShapeLocationMean { get; private set; }

        public GammaArray1D ShapeLocationPrecision { get; private set; }
            
        public GaussianArray3D ShapePartOffsetWeights { get; private set; }

        public GaussianArray3D ShapePartLogScaleWeights { get; private set; }

        public GaussianArray2D ShapePartAngleWeights { get; private set; }

        public GammaArray2D ShapePartOffsetPrecisions { get; private set; }

        public GammaArray2D ShapePartLogScalePrecisions { get; private set; }

        public GammaArray1D ShapePartAnglePrecisions { get; private set; }

        public static ShapeModel Load(string filename)
        {
            using (Stream stream = new FileStream(filename, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                return (ShapeModel) new BinaryFormatter().Deserialize(stream);
            }
        }
        
        public void Save(string filename)
        {
            using (Stream stream = new FileStream(filename, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                new BinaryFormatter().Serialize(stream, this);
            }
        }

        public ShapeModelSample[] Sample(double[][] traits, bool withNoise, bool ignoreLocationPrior)
        {
            Debug.Assert(traits[0].Length == this.TraitCount);

            int sampleCount = traits.Length;

            ShapeModelSample[] result = new ShapeModelSample[sampleCount];

            double[] shapeLocationMean = this.ShapeLocationMean.Sample();
            double[] shapeLocationPrecision = this.ShapeLocationPrecision.Sample();
            double[][][] shapePartOffsetMeanWeights = GetMean(this.ShapePartOffsetWeights); //this.ShapePartOffsetWeights.Sample();
            double[][][] shapePartLogScaleMeanWeights = GetMean(this.ShapePartLogScaleWeights);//this.ShapePartScaleWeights.Sample();
            double[][] shapePartAngleMeanWeights = GetMean(this.ShapePartAngleWeights);//this.ShapePartAngleWeights.Sample();
            double[][] shapePartOffsetPrecisions = GetMean(this.ShapePartOffsetPrecisions);//this.ShapePartOffsetPrecisions.Sample();
            double[][] shapePartLogScalePrecisions = GetMean(this.ShapePartLogScalePrecisions);//this.ShapePartScalePrecisions.Sample();
            double[] shapePartAnglePrecisions = GetMean(this.ShapePartAnglePrecisions);//this.ShapePartAnglePrecisions.Sample();

            for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
            {
                // Sample locations from priors
                double locationX, locationY;
                if (ignoreLocationPrior)
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
                result[sampleIndex] = new ShapeModelSample() { ShapeModel = this, ShapePartLocations = new Vector[this.ShapePartCount], ShapePartOrientations = new PositiveDefiniteMatrix[this.ShapePartCount] };

                for (int shapePartIndex = 0; shapePartIndex < this.ShapePartCount; ++shapePartIndex)
                {
                    Vector traitVector = Vector.FromArray(traits[sampleIndex]);
                    double meanOffsetX = Vector.InnerProduct(traitVector, Vector.FromArray(shapePartOffsetMeanWeights[shapePartIndex][0]));
                    double meanOffsetY = Vector.InnerProduct(traitVector, Vector.FromArray(shapePartOffsetMeanWeights[shapePartIndex][1]));
                    double meanLogScaleX = Vector.InnerProduct(traitVector, Vector.FromArray(shapePartLogScaleMeanWeights[shapePartIndex][0]));
                    double meanLogScaleY = Vector.InnerProduct(traitVector, Vector.FromArray(shapePartLogScaleMeanWeights[shapePartIndex][1]));
                    double meanAngle = Vector.InnerProduct(traitVector, Vector.FromArray(shapePartAngleMeanWeights[shapePartIndex]));

                    double x = locationX + Gaussian.FromMeanAndPrecision(meanOffsetX, shapePartOffsetPrecisions[shapePartIndex][0]).Sample();
                    double y = locationY + Gaussian.FromMeanAndPrecision(meanOffsetY, shapePartOffsetPrecisions[shapePartIndex][1]).Sample();
                    double logScaleX = Gaussian.FromMeanAndPrecision(meanLogScaleX, shapePartLogScalePrecisions[shapePartIndex][0]).Sample();
                    double logScaleY = Gaussian.FromMeanAndPrecision(meanLogScaleY, shapePartLogScalePrecisions[shapePartIndex][1]).Sample();
                    double angle = Gaussian.FromMeanAndPrecision(meanAngle, shapePartAnglePrecisions[shapePartIndex]).Sample();

                    // Truncate samples to match the model constraints (not entirely correct)
                    angle = Math.Min(Math.PI / 4 - 0.01, angle);
                    angle = Math.Max(-Math.PI / 4 + 0.01, angle);

                    PositiveDefiniteMatrix orientation = ShapeFactors.MatrixFromAngleScale(logScaleX, logScaleY, angle);

                    result[sampleIndex].ShapePartLocations[shapePartIndex] = Vector.FromArray(x, y);
                    result[sampleIndex].ShapePartOrientations[shapePartIndex] = orientation;
                }

                var labels = new bool[this.GridWidth, this.GridHeight];
                var noisyLabels = new bool[this.GridWidth, this.GridHeight];
                for (int i = 0; i < this.GridWidth; ++i)
                {
                    for (int j = 0; j < this.GridHeight; ++j)
                    {
                        labels[i, j] = false;
                        for (int k = 0; k < this.ShapePartCount; ++k)
                        {
                            labels[i, j] |= ShapeFactors.LabelFromShape(
                                Vector.FromArray((i + 0.5) / this.GridWidth, (j + 0.5) / this.GridHeight),
                                result[sampleIndex].ShapePartLocations[k][0],
                                result[sampleIndex].ShapePartLocations[k][1],
                                result[sampleIndex].ShapePartOrientations[k]);
                        }

                        noisyLabels[i, j] = Rand.Double() >= this.ObservationNoiseProb ? labels[i, j] : !labels[i, j];
                    }
                }

                result[sampleIndex].Labels = withNoise ? noisyLabels : labels;
            }

            return result;
        }

        public ShapeModelSample[] Sample(int sampleCount, bool withNoise, bool withSymmetryBreaking, bool ignoreLocationPrior)
        {
            double[][] traits = new double[sampleCount][];
            for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
            {
                traits[sampleIndex] = new double[this.TraitCount];
                for (int traitIndex = 0; traitIndex < this.TraitCount; ++traitIndex)
                {
                    if (traitIndex == traits.Length - 1)
                    {
                        traits[sampleIndex][traitIndex] = 1;
                    }
                    else if (withSymmetryBreaking && traitIndex < sampleCount)
                    {
                        traits[sampleIndex][traitIndex] = traitIndex == sampleIndex ? 1 : 0;
                    }
                    else
                    {
                        traits[sampleIndex][traitIndex] = Gaussian.Sample(0, 1);
                    }
                }
            }

            return this.Sample(traits, withNoise, ignoreLocationPrior);
        }

        private static double[][][] GetMean(GaussianArray3D dist)
        {
            return Util.ArrayInit(dist.Count, i => Util.ArrayInit(dist[i].Count, j => Util.ArrayInit(dist[i][j].Count, k => dist[i][j][k].GetMean())));
        }

        private static double[][] GetMean(GaussianArray2D dist)
        {
            return Util.ArrayInit(dist.Count, i => Util.ArrayInit(dist[i].Count, j => dist[i][j].GetMean()));
        }

        private static double[][] GetMean(GammaArray2D dist)
        {
            return Util.ArrayInit(dist.Count, i => Util.ArrayInit(dist[i].Count, j => dist[i][j].GetMean()));
        }

        private static double[] GetMean(GammaArray1D dist)
        {
            return Util.ArrayInit(dist.Count, i => dist[i].GetMean());
        }
    }

    public class ShapeFittingInfo
    {
        public ShapeFittingInfo(
            GaussianArray2D shapeTraits,
            GaussianArray3D shapePartLocations,
            WishartArray2D shapePartOrientations,
            GaussianArray1D globalLogScales)
        {
            this.ShapeTraits = shapeTraits;
            this.ShapePartLocations = shapePartLocations;
            this.ShapePartOrientations = shapePartOrientations;
            this.GlobalLogScales = globalLogScales;
        }
        
        public GaussianArray2D ShapeTraits { get; private set; }

        public GaussianArray3D ShapePartLocations { get; private set; }

        public WishartArray2D ShapePartOrientations { get; private set; }

        public GaussianArray1D GlobalLogScales { get; private set; }
    }

    public class ShapeModelLearningProgressEventArgs : EventArgs
    {
        public ShapeModelLearningProgressEventArgs(
            int iterationsCompleted,
            ShapeModel shapeModel,
            ShapeFittingInfo fittingInfo)
        {
            this.IterationsCompleted = iterationsCompleted;
            this.ShapeModel = shapeModel;
            this.FittingInfo = fittingInfo;
        }
        
        public int IterationsCompleted { get; private set; }
        
        public ShapeModel ShapeModel { get; private set; }

        public ShapeFittingInfo FittingInfo { get; private set; }
    }

    public class MaskCompletionProgressEventArgs : EventArgs
    {
        public MaskCompletionProgressEventArgs(int iterationsCompleted, LabelArray completedMasks)
        {
            this.IterationsCompleted = iterationsCompleted;
            this.CompletedMasks = completedMasks;
        }

        public int IterationsCompleted { get; private set; }

        public LabelArray CompletedMasks { get; private set; }
    }

    public class ShapeModelInference
    {
        public const double MeanEllipseHalfSize = 0.2;

        #region Model variables

        // Ranges

        protected Variable<int> observationCount;

        protected Variable<int> gridWidth;
        
        protected Variable<int> gridHeight;

        protected Variable<int> shapePartCount;

        protected Variable<int> traitCount;
        
        protected Range observationRange;

        protected Range xyRange;

        protected Range widthRange;

        protected Range heightRange;

        protected Range shapePartRange;

        protected Range traitRange;

        // Shape model

        protected Variable<GaussianArray3D> shapePartOffsetWeightPriors;

        protected Variable<GaussianArray3D> shapePartLogScaleWeightPriors;

        protected Variable<GaussianArray2D> shapePartAngleWeightPriors;
        
        // range order: shape part, xy, trait
        protected VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> shapePartOffsetWeights;

        // range order: shape part, xy, trait
        protected VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> shapePartLogScaleWeights;

        // range order: shape part, trait
        protected VariableArray<VariableArray<double>, double[][]> shapePartAngleWeights;

        protected Variable<GaussianArray1D> globalLogScalePrior;

        protected VariableArray<double> globalLogScale;
        
        protected Variable<GaussianArray1D> shapeLocationMeanPrior;
        
        protected Variable<GammaArray1D> shapeLocationPrecisionPrior;
        
        protected VariableArray<double> shapeLocationMean;
        
        protected VariableArray<double> shapeLocationPrecision;
        
        protected VariableArray<VariableArray<double>, double[][]> shapeLocation;
        
        protected VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> shapePartLocation;
        
        protected VariableArray<VariableArray<PositiveDefiniteMatrix>, PositiveDefiniteMatrix[][]> shapePartOrientation;
        
        protected Variable<GammaArray2D> shapePartOffsetPrecisionPriors;

        protected Variable<GammaArray2D> shapePartLogScalePrecisionPriors;

        protected Variable<GammaArray1D> shapePartAnglePrecisionPriors;
        
        protected VariableArray<VariableArray<double>, double[][]> shapePartOffsetPrecisions;

        protected VariableArray<VariableArray<double>, double[][]> shapePartLogScalePrecisions;

        protected VariableArray<double> shapePartAnglePrecisions;
        
        protected Variable<GaussianArray2D> shapeTraitsPrior;
        
        protected VariableArray<VariableArray<double>, double[][]> shapeTraits;

        // Observation noise model

        protected Variable<double> observationNoiseProbability { get; private set; }

        // Grid specification

        protected VariableArray2D<Vector> pixelCoords;

        protected VariableArray<VariableArray2D<bool>, bool[][,]> labels;

        protected VariableArray<VariableArray2D<bool>, bool[][,]> noisyLabels;

        protected VariableArray<VariableArray2D<Bernoulli>, Bernoulli[][,]> noisyLabelsConstraint;

        #endregion

        #region State

        private bool initialized;

        private bool working;

        #endregion

        public ShapeModelInference()
        {
            this.TrainingIterationCount = 1000;
            this.MaskCompletionIterationCount = 1000;
            this.IterationsBetweenCallbacks = 100;
            
            this.DefineModel();
        }

        public int TrainingIterationCount { get; set; }

        public int MaskCompletionIterationCount { get; set; }

        public int IterationsBetweenCallbacks { get; set; }

        public event EventHandler<ShapeModelLearningProgressEventArgs> ShapeModelLearningProgress;

        public event EventHandler<MaskCompletionProgressEventArgs> MaskCompletionProgress;

        public void SetPriorBeliefs(int gridWidth, int gridHeight, int traitCount, int shapePartCount, double observationNoiseProb)
        {
            this.gridWidth.ObservedValue = gridWidth;
            this.gridHeight.ObservedValue = gridHeight;
            this.traitCount.ObservedValue = traitCount;
            this.shapePartCount.ObservedValue = shapePartCount;
            this.observationNoiseProbability.ObservedValue = observationNoiseProb;

            double traitInvScale = 1.0 / (traitCount - 1);
            Gaussian offsetMeanPrior = Gaussian.FromMeanAndVariance(0, 1);// Gaussian.FromMeanAndVariance(0, 0.3 * 0.3);
            Gaussian offsetAdjustmentPrior =  Gaussian.FromMeanAndVariance(0, 1); //Gaussian.FromMeanAndVariance(0, 0.05 * 0.05 * traitInvScale);
            Gaussian logScaleMeanPrior = Gaussian.FromMeanAndVariance(0, 3 * 3);// Gaussian.FromMeanAndVariance(-1.5, 1);
            Gaussian logScaleAdjustmentPrior = Gaussian.FromMeanAndVariance(0, 3 * 3);// Gaussian.FromMeanAndVariance(0, 0.15 * 0.15 * traitInvScale);
            Gaussian angleMeanPrior = Gaussian.FromMeanAndVariance(0, 1);// Gaussian.FromMeanAndVariance(0, 10);
            Gaussian angleAdjustmentPrior = Gaussian.FromMeanAndVariance(0, 1);// Gaussian.FromMeanAndVariance(0, Math.PI * Math.PI / (18 * 18) * traitInvScale);
            
            this.shapeLocationMeanPrior.ObservedValue = CreateGaussianArray1D(Gaussian.FromMeanAndVariance(0.5, 0.3 * 0.3), 2);
            this.shapeLocationPrecisionPrior.ObservedValue = CreateGammaArray1D(Gamma.FromMeanAndVariance(1.0 / (0.3 * 0.3), 10 * 10), 2);
            this.shapePartOffsetWeightPriors.ObservedValue = CreateGaussianArray3D((i, j, k) => k == traitCount - 1 ? offsetMeanPrior : offsetAdjustmentPrior, shapePartCount, 2, traitCount);
            this.shapePartLogScaleWeightPriors.ObservedValue = CreateGaussianArray3D((i, j, k) => k == traitCount - 1 ? logScaleMeanPrior : logScaleAdjustmentPrior, shapePartCount, 2, traitCount);
            this.shapePartAngleWeightPriors.ObservedValue = CreateGaussianArray2D((i, j) => j == traitCount - 1 ? angleMeanPrior : angleAdjustmentPrior, shapePartCount, traitCount);
            this.shapePartOffsetPrecisionPriors.ObservedValue = CreateGammaArray2D(Gamma.PointMass(1.0 / (0.005 * 0.005)), shapePartCount, 2);
            this.shapePartLogScalePrecisionPriors.ObservedValue = CreateGammaArray2D(Gamma.PointMass(1.0 / ((0.005) * 0.005)), shapePartCount, 2);
            this.shapePartAnglePrecisionPriors.ObservedValue = CreateGammaArray1D(Gamma.PointMass(1.0 / (Math.PI * 0.01 * Math.PI * 0.01)), shapePartCount);
            
            this.pixelCoords.ObservedValue = Util.ArrayInit(gridWidth, gridHeight, (i, j) => Vector.FromArray((i + 0.5) / gridWidth, (j + 0.5) / gridHeight));

            this.initialized = true;
        }

        public void SetBeliefs(ShapeModel shapeModel)
        {
            this.SetPriorBeliefs(shapeModel.GridWidth, shapeModel.GridHeight, shapeModel.TraitCount, shapeModel.ShapePartCount, shapeModel.ObservationNoiseProb);

            this.shapeLocationMeanPrior.ObservedValue = shapeModel.ShapeLocationMean;
            this.shapeLocationPrecisionPrior.ObservedValue = shapeModel.ShapeLocationPrecision;
            this.shapePartOffsetWeightPriors.ObservedValue = shapeModel.ShapePartOffsetWeights;
            this.shapePartLogScaleWeightPriors.ObservedValue = shapeModel.ShapePartLogScaleWeights;
            this.shapePartAngleWeightPriors.ObservedValue = shapeModel.ShapePartAngleWeights;
            this.shapePartOffsetPrecisionPriors.ObservedValue = shapeModel.ShapePartOffsetPrecisions;
            this.shapePartLogScalePrecisionPriors.ObservedValue = shapeModel.ShapePartLogScalePrecisions;
            this.shapePartAnglePrecisionPriors.ObservedValue = shapeModel.ShapePartAnglePrecisions;
        }

        public Tuple<ShapeModel, ShapeFittingInfo> LearnModel(bool[][,] shapeMasks, bool breakTraitSymmetry = true)
        {
            Debug.Assert(this.initialized);
            Debug.Assert(!this.working);
            Debug.Assert(shapeMasks.All(mask => mask.GetLength(0) == this.gridWidth.ObservedValue && mask.GetLength(1) == this.gridHeight.ObservedValue));

            this.working = true;
            
            this.observationCount.ObservedValue = shapeMasks.Length;
            this.noisyLabels.ObservedValue = shapeMasks;

            this.InitializePerObservation();

            // Setup traits prior to break symmetry
            if (breakTraitSymmetry)
            {
                int squarePartSize = Math.Min(this.observationCount.ObservedValue, this.traitCount.ObservedValue - 1);
                for (int i = 0; i < squarePartSize; ++i)
                {
                    for (int j = 0; j < squarePartSize; ++j)
                    {
                        this.shapeTraitsPrior.ObservedValue[i][j] = i == j ? Gaussian.PointMass(1) : Gaussian.PointMass(0);
                    }
                }
            }

            var engine = new InferenceEngine();
            engine.Compiler.RequiredQuality = QualityBand.Unknown;
            engine.Compiler.RecommendedQuality = QualityBand.Unknown;
            //engine.Compiler.UseParallelForLoops = true;
            engine.OptimiseForVariables = new IVariable[]
            {
                this.shapeLocationMean,
                this.shapeLocationPrecision,
                this.shapePartLocation,
                this.shapePartOrientation,
                this.shapePartOffsetWeights,
                this.shapePartLogScaleWeights,
                this.shapePartAngleWeights,
                this.shapePartOffsetPrecisions,
                this.shapePartLogScalePrecisions,
                this.shapePartAnglePrecisions,
                this.shapeTraits,
                this.globalLogScale,
            };

            for (int iteration = this.IterationsBetweenCallbacks; iteration <= this.TrainingIterationCount; iteration += this.IterationsBetweenCallbacks)
            {
                engine.NumberOfIterations = iteration;
                ShapeModel model = InferShapeModel(engine);
                ShapeFittingInfo fittingInfo = InferShapeFitting(engine);

                if (this.ShapeModelLearningProgress != null)
                {
                    this.ShapeModelLearningProgress(this, new ShapeModelLearningProgressEventArgs(iteration, model, fittingInfo));
                }
            }

            engine.NumberOfIterations = this.TrainingIterationCount;
            ShapeModel resultingModel = InferShapeModel(engine);
            ShapeFittingInfo resultingFittingInfo = InferShapeFitting(engine);

            this.working = false;
            return Tuple.Create(resultingModel, resultingFittingInfo);
        }

        public LabelArray CompleteMasks(bool[][,] shapeMasks, bool[][,] availableMaskPixels)
        {
            Debug.Assert(this.initialized);
            Debug.Assert(!this.working);
            Debug.Assert(shapeMasks.Length == availableMaskPixels.Length);
            Debug.Assert(shapeMasks.All(mask => mask.GetLength(0) == this.gridWidth.ObservedValue && mask.GetLength(1) == this.gridHeight.ObservedValue));
            Debug.Assert(availableMaskPixels.All(mask => mask.GetLength(0) == this.gridWidth.ObservedValue && mask.GetLength(1) == this.gridHeight.ObservedValue));

            this.working = true;

            this.observationCount.ObservedValue = shapeMasks.Length;

            this.InitializePerObservation();
            
            for (int i = 0; i < shapeMasks.Length; ++i)
            {
                for (int x = 0; x < this.gridWidth.ObservedValue; ++x)
                {
                    for (int y = 0; y < this.gridHeight.ObservedValue; ++y)
                    {
                        if (availableMaskPixels[i][x, y])
                        {
                            this.noisyLabelsConstraint.ObservedValue[i][x, y] = Bernoulli.PointMass(shapeMasks[i][x, y]);
                        }
                    }
                }
            }

            var engine = new InferenceEngine();
            engine.Compiler.RequiredQuality = QualityBand.Unknown;
            engine.Compiler.RecommendedQuality = QualityBand.Unknown;
            //engine.Compiler.UseParallelForLoops = true;
            engine.OptimiseForVariables = new IVariable[]
            {
                this.labels,
            };

            for (int iteration = this.IterationsBetweenCallbacks; iteration <= this.MaskCompletionIterationCount; iteration += this.IterationsBetweenCallbacks)
            {
                engine.NumberOfIterations = iteration;
                LabelArray completedLabels = engine.Infer<LabelArray>(this.labels);

                if (this.MaskCompletionProgress != null)
                {
                    this.MaskCompletionProgress(this, new MaskCompletionProgressEventArgs(iteration, completedLabels));
                }
            }

            engine.NumberOfIterations = this.MaskCompletionIterationCount;
            var result = engine.Infer<LabelArray>(this.labels);

            this.working = false;
            return result;
        }

        private ShapeModel InferShapeModel(InferenceEngine engine)
        {
            return new ShapeModel(
                this.gridWidth.ObservedValue,
                this.gridHeight.ObservedValue,
                this.observationNoiseProbability.ObservedValue,
                engine.Infer<GaussianArray1D>(this.shapeLocationMean),
                engine.Infer<GammaArray1D>(this.shapeLocationPrecision),
                engine.Infer<GaussianArray3D>(this.shapePartOffsetWeights),
                engine.Infer<GaussianArray3D>(this.shapePartLogScaleWeights),
                engine.Infer<GaussianArray2D>(this.shapePartAngleWeights),
                engine.Infer<GammaArray2D>(this.shapePartOffsetPrecisions),
                engine.Infer<GammaArray2D>(this.shapePartLogScalePrecisions),
                engine.Infer<GammaArray1D>(this.shapePartAnglePrecisions));
        }

        private ShapeFittingInfo InferShapeFitting(InferenceEngine engine)
        {
            return new ShapeFittingInfo(
                engine.Infer<GaussianArray2D>(this.shapeTraits),
                engine.Infer<GaussianArray3D>(this.shapePartLocation),
                engine.Infer<WishartArray2D>(this.shapePartOrientation),
                engine.Infer<GaussianArray1D>(this.globalLogScale));
        }

        private void DefineModel()
        {
            this.observationCount = Variable.New<int>().Named("observation_count");
            this.gridWidth = Variable.New<int>().Named("grid_width");
            this.gridHeight = Variable.New<int>().Named("grid_height");
            this.shapePartCount = Variable.New<int>().Named("shape_part_count");
            this.traitCount = Variable.New<int>().Named("trait_count");

            this.observationRange = new Range(this.observationCount).Named("observation_range");
            this.xyRange = new Range(2).Named("xy_range");
            this.widthRange = new Range(this.gridWidth).Named("width_range");
            this.heightRange = new Range(this.gridHeight).Named("height_range");
            this.shapePartRange = new Range(this.shapePartCount).Named("shape_part_range");
            this.traitRange = new Range(this.traitCount).Named("trait_range");

            this.shapeLocationMeanPrior = Variable.New<GaussianArray1D>().Named("shape_location_mean_prior");
            this.shapeLocationMean = Variable.Array<double>(this.xyRange).Named("shape_location_mean");
            this.shapeLocationMean.SetTo(Variable<double[]>.Random(this.shapeLocationMeanPrior));

            this.shapeLocationPrecisionPrior = Variable.New<GammaArray1D>().Named("shape_location_prec_prior");
            this.shapeLocationPrecision = Variable.Array<double>(this.xyRange).Named("shape_location_prec");
            this.shapeLocationPrecision.SetTo(Variable<double[]>.Random(this.shapeLocationPrecisionPrior));

            this.shapePartOffsetWeightPriors = Variable.New<GaussianArray3D>().Named("shape_part_offset_weight_prior");
            this.shapePartOffsetWeights = Variable.Array(Variable.Array(Variable.Array<double>(this.traitRange), this.xyRange), this.shapePartRange).Named("shape_part_offset_weights");
            this.shapePartOffsetWeights.SetTo(Variable<double[][][]>.Random(this.shapePartOffsetWeightPriors));

            this.shapePartLogScaleWeightPriors = Variable.New<GaussianArray3D>().Named("shape_part_scale_weight_prior");
            this.shapePartLogScaleWeights = Variable.Array(Variable.Array(Variable.Array<double>(this.traitRange), this.xyRange), this.shapePartRange).Named("shape_part_scale_weights");
            this.shapePartLogScaleWeights.SetTo(Variable<double[][][]>.Random(this.shapePartLogScaleWeightPriors));

            this.shapePartAngleWeightPriors = Variable.New<GaussianArray2D>().Named("shape_part_angle_weight_prior");
            this.shapePartAngleWeights = Variable.Array(Variable.Array<double>(this.traitRange), this.shapePartRange).Named("shape_part_angle_weights");
            this.shapePartAngleWeights.SetTo(Variable<double[][]>.Random(this.shapePartAngleWeightPriors));

            this.shapePartOffsetPrecisionPriors = Variable.New<GammaArray2D>().Named("shape_part_offset_prec_prior");
            this.shapePartOffsetPrecisions = Variable.Array(Variable.Array<double>(this.xyRange), this.shapePartRange).Named("shape_part_offset_prec");
            this.shapePartOffsetPrecisions.SetTo(Variable<double[][]>.Random(this.shapePartOffsetPrecisionPriors));

            this.shapePartLogScalePrecisionPriors = Variable.New<GammaArray2D>().Named("shape_part_scale_prec_prior");
            this.shapePartLogScalePrecisions = Variable.Array(Variable.Array<double>(this.xyRange), this.shapePartRange).Named("shape_part_scale_prec");
            this.shapePartLogScalePrecisions.SetTo(Variable<double[][]>.Random(this.shapePartLogScalePrecisionPriors));

            this.shapePartAnglePrecisionPriors = Variable.New<GammaArray1D>().Named("shape_part_angle_prec_prior");
            this.shapePartAnglePrecisions = Variable.Array<double>(this.shapePartRange).Named("shape_part_angle_prec");
            this.shapePartAnglePrecisions.SetTo(Variable<double[]>.Random(this.shapePartAnglePrecisionPriors));

            this.globalLogScalePrior = Variable.New<GaussianArray1D>().Named("global_log_scale_prior");
            this.globalLogScale = Variable.Array<double>(this.observationRange).Named("global_log_scale");
            this.globalLogScale.SetTo(Variable<double[]>.Random(this.globalLogScalePrior));
            
            this.shapeLocation = Variable.Array(Variable.Array<double>(this.xyRange), this.observationRange).Named("shape_location");

            this.shapePartLocation = Variable.Array(Variable.Array(Variable.Array<double>(this.xyRange), this.shapePartRange), this.observationRange).Named("shape_part_location");
            this.shapePartLocation.AddAttribute(new PointEstimate());

            this.shapePartOrientation = Variable.Array(Variable.Array<PositiveDefiniteMatrix>(this.shapePartRange), this.observationRange).Named("shape_part_orientation");
            this.shapePartOrientation.AddAttribute(new PointEstimate());

            this.shapeTraitsPrior = Variable.New<GaussianArray2D>().Named("shape_traits_prior"); // Needs to be observed in the derived classes
            this.shapeTraits = Variable.Array(Variable.Array<double>(this.traitRange), this.observationRange).Named("shape_traits");
            this.shapeTraits.SetTo(Variable<double[][]>.Random(this.shapeTraitsPrior));

            this.observationNoiseProbability = Variable.New<double>().Named("observation_noise_prob");

            this.pixelCoords = Variable.Array<Vector>(this.widthRange, this.heightRange).Named("pixel_coords");

            this.labels =
                Variable.Array<VariableArray2D<bool>, bool[][,]>(Variable.Array<bool>(this.widthRange, this.heightRange), this.observationRange)
                        .Named("labels");
            this.noisyLabels =
                Variable.Array<VariableArray2D<bool>, bool[][,]>(Variable.Array<bool>(this.widthRange, this.heightRange), this.observationRange)
                        .Named("noisy_labels");

            this.noisyLabelsConstraint =
                Variable.Array<VariableArray2D<Bernoulli>, Bernoulli[][,]>(Variable.Array<Bernoulli>(this.widthRange, this.heightRange), this.observationRange)
                        .Named("noisy_labels_constraint");

            using (var observationIter = Variable.ForEach(this.observationRange))
            {
                this.shapeLocation[this.observationRange][this.xyRange] = Variable.GaussianFromMeanAndPrecision(this.shapeLocationMean[this.xyRange], this.shapeLocationPrecision[this.xyRange]);
                
                using (Variable.ForEach(this.shapePartRange))
                {
                    const double productDamping = 0.5;
                    
                    // Location

                    var shapePartOffsetMeanTraitWeightProducts = Variable.Array(Variable.Array<double>(this.traitRange), this.xyRange).Named("shape_part_offset_mean_products");
                    shapePartOffsetMeanTraitWeightProducts[this.xyRange][this.traitRange] =
                        Variable<double>.Factor(Factor.Product_SHG09, this.shapeTraits[this.observationRange][this.traitRange], this.shapePartOffsetWeights[this.shapePartRange][this.xyRange][this.traitRange]);
                    var shapePartOffsetMeanTraitWeightProductsDamped = Variable.Array(Variable.Array<double>(this.traitRange), this.xyRange).Named("shape_part_offset_mean_products_damped");
                    shapePartOffsetMeanTraitWeightProductsDamped[this.xyRange][this.traitRange] = Variable<double>.Factor(Damp.Forward<double>, shapePartOffsetMeanTraitWeightProducts[this.xyRange][this.traitRange], productDamping);

                    var shapePartOffsetMean = Variable.Array<double>(this.xyRange).Named("shape_part_offset_mean");
                    shapePartOffsetMean[this.xyRange] = Variable.Sum(shapePartOffsetMeanTraitWeightProductsDamped[this.xyRange]);
                    var shapePartOffset = Variable.GaussianFromMeanAndPrecision(
                        shapePartOffsetMean[this.xyRange], this.shapePartOffsetPrecisions[this.shapePartRange][this.xyRange]).Named("shape_part_offset");
                    
                    this.shapePartLocation[this.observationRange][this.shapePartRange][this.xyRange] = this.shapeLocation[this.observationRange][this.xyRange] + shapePartOffset;

                    // Orientation

                    var shapePartLogScaleMeanTraitWeightProducts = Variable.Array(Variable.Array<double>(this.traitRange), this.xyRange).Named("shape_part_logscale_mean_products");
                    shapePartLogScaleMeanTraitWeightProducts[this.xyRange][this.traitRange] =
                        Variable<double>.Factor(Factor.Product_SHG09, this.shapeTraits[this.observationRange][this.traitRange], this.shapePartLogScaleWeights[this.shapePartRange][this.xyRange][this.traitRange]);
                    var shapePartLogScaleMeanTraitWeightProductsDamped = Variable.Array(Variable.Array<double>(this.traitRange), this.xyRange).Named("shape_part_logscale_mean_products_damped");
                    shapePartLogScaleMeanTraitWeightProductsDamped[this.xyRange][this.traitRange] = Variable<double>.Factor(Damp.Forward<double>, shapePartLogScaleMeanTraitWeightProducts[this.xyRange][this.traitRange], productDamping);

                    var shapePartLogScaleMean = Variable.Array<double>(this.xyRange).Named("shape_part_logscale_mean");
                    shapePartLogScaleMean[this.xyRange] = Variable.Sum(shapePartLogScaleMeanTraitWeightProductsDamped[this.xyRange]);
                    var shapePartLogScale = Variable.Array<double>(this.xyRange).Named("shape_part_logscale");
                    shapePartLogScale[this.xyRange] = Variable.GaussianFromMeanAndPrecision(
                        shapePartLogScaleMean[this.xyRange], this.shapePartLogScalePrecisions[this.shapePartRange][this.xyRange]);

                    var shapePartAngleMeanTraitWeightProducts = Variable.Array<double>(this.traitRange).Named("shape_part_angle_mean_products");
                    shapePartAngleMeanTraitWeightProducts[this.traitRange] =
                        Variable<double>.Factor(Factor.Product_SHG09, this.shapeTraits[this.observationRange][this.traitRange], this.shapePartAngleWeights[this.shapePartRange][this.traitRange]);
                    var shapePartAngleMeanTraitWeightProductsDamped = Variable.Array<double>(this.traitRange).Named("shape_part_angle_mean_products_damped");
                    shapePartAngleMeanTraitWeightProductsDamped[this.traitRange] = Variable<double>.Factor(Damp.Forward<double>, shapePartAngleMeanTraitWeightProducts[this.traitRange], productDamping);

                    var shapePartAngleMean = Variable.Sum(shapePartAngleMeanTraitWeightProductsDamped).Named("shape_part_angle_mean");
                    var shapePartAngle = Variable.GaussianFromMeanAndPrecision(
                        shapePartAngleMean, this.shapePartAnglePrecisions[this.shapePartRange]).Named("shape_part_angle");

                    this.shapePartOrientation[this.observationRange][this.shapePartRange] = Variable<PositiveDefiniteMatrix>.Factor(
                        ShapeFactors.MatrixFromAngleScale, shapePartLogScale[0] /*+ this.globalLogScale[observationRange]*/, shapePartLogScale[1] /*+ this.globalLogScale[observationRange]*/, shapePartAngle);
                    this.shapePartOrientation[this.observationRange][this.shapePartRange].AddAttribute(new MarginalPrototype(new Wishart(2)));
                }

                using (Variable.ForEach(this.widthRange))
                using (Variable.ForEach(this.heightRange))
                {
                    var labelsByPart = Variable.Array<bool>(this.shapePartRange).Named("labels_by_part");

                    using (Variable.ForEach(this.shapePartRange))
                    {
                       labelsByPart[this.shapePartRange] = Variable<bool>.Factor(
                            ShapeFactors.LabelFromShape,
                            this.pixelCoords[this.widthRange, this.heightRange],
                            this.shapePartLocation[this.observationRange][this.shapePartRange][0],
                            this.shapePartLocation[this.observationRange][this.shapePartRange][1],
                            this.shapePartOrientation[this.observationRange][this.shapePartRange]);
                    }

                    this.labels[this.observationRange][this.widthRange, this.heightRange] = Variable<bool>.Factor(Factors.AnyTrue, labelsByPart);

                    //using (Variable.Repeat(100))
                    {
                        using (Variable.If(this.labels[this.observationRange][this.widthRange, this.heightRange]))
                        {
                            this.noisyLabels[this.observationRange][this.widthRange, this.heightRange] =
                                !Variable.Bernoulli(this.observationNoiseProbability);
                        }

                        using (Variable.IfNot(this.labels[this.observationRange][this.widthRange, this.heightRange]))
                        {
                            this.noisyLabels[this.observationRange][this.widthRange, this.heightRange] =
                                Variable.Bernoulli(this.observationNoiseProbability);
                        }
                    }

                    //Variable.ConstrainEqualRandom(
                    //    this.noisyLabels[this.observationRange][this.widthRange, this.heightRange],
                    //    this.noisyLabelsConstraint[this.observationRange][this.widthRange, this.heightRange]);
                }
            }
        }

        private void InitializePerObservation()
        {
            var partLocationInit = Util.ArrayInit(this.shapePartCount.ObservedValue, 2, (i, j) => Gaussian.PointMass(0.49 + 0.02 * Rand.Double()));
            this.shapePartLocation.InitialiseTo(
                CreateGaussianArray3D((i, j, k) => partLocationInit[j, k], this.observationCount.ObservedValue, this.shapePartCount.ObservedValue, 2));

            var partOrientationInit = Util.ArrayInit(
                this.shapePartCount.ObservedValue, i => Wishart.PointMass(new PositiveDefiniteMatrix(new double[,]{ { 0.99 + Rand.Double() * 0.02, 0 }, { 0, 0.99 + Rand.Double() * 0.02 } }) * (1.0 / (MeanEllipseHalfSize * MeanEllipseHalfSize))));
            this.shapePartOrientation.InitialiseTo(
                CreateWishartArray2D((i, j) => partOrientationInit[j], this.observationCount.ObservedValue, this.shapePartCount.ObservedValue));

            // Initialized here because we might want to break scale symmetry
            this.globalLogScalePrior.ObservedValue = CreateGaussianArray1D(
                i => i < this.traitCount.ObservedValue - 1 ? Gaussian.PointMass(0) : Gaussian.FromMeanAndVariance(0, 1),
                this.observationCount.ObservedValue);
            
            this.noisyLabelsConstraint.ObservedValue = Util.ArrayInit(this.observationCount.ObservedValue, o => Util.ArrayInit(this.gridWidth.ObservedValue, this.gridHeight.ObservedValue, (i, j) => Bernoulli.Uniform())); // No constraint by default

            this.shapeTraitsPrior.ObservedValue = new GaussianArray2D(this.observationCount.ObservedValue);
            for (int i = 0; i < this.observationCount.ObservedValue; ++i)
            {
                this.shapeTraitsPrior.ObservedValue[i] = new GaussianArray1D(this.traitCount.ObservedValue);
                for (int j = 0; j < this.traitCount.ObservedValue; ++j)
                {
                    if (j == this.traitCount.ObservedValue - 1)
                    {
                        this.shapeTraitsPrior.ObservedValue[i][j] = Gaussian.PointMass(1);
                    }
                    else
                    {
                        this.shapeTraitsPrior.ObservedValue[i][j] = Gaussian.FromMeanAndVariance(0, 1);
                    }
                }
            }
        }

        private static GaussianArray1D CreateGaussianArray1D(Gaussian dist, int length)
        {
            return CreateGaussianArray1D(i => dist, length);
        }

        private static GaussianArray1D CreateGaussianArray1D(Func<int, Gaussian> dist, int length)
        {
            return new GaussianArray1D(Util.ArrayInit(length, i => dist(i)));
        }

        private static GaussianArray2D CreateGaussianArray2D(Gaussian dist, int length1, int length2)
        {
            return CreateGaussianArray2D((i, j) => dist, length1, length2);
        }

        private static GaussianArray2D CreateGaussianArray2D(Func<int, int, Gaussian> dist, int length1, int length2)
        {
            return new GaussianArray2D(Util.ArrayInit(length1, i => CreateGaussianArray1D(j => dist(i, j), length2)));
        }

        private static GaussianArray3D CreateGaussianArray3D(Gaussian dist, int length1, int length2, int length3)
        {
            return CreateGaussianArray3D((i, j, k) => dist, length1, length2, length3);
        }

        private static GaussianArray3D CreateGaussianArray3D(Func<int, int, int, Gaussian> dist, int length1, int length2, int length3)
        {
            return new GaussianArray3D(Util.ArrayInit(length1, i => CreateGaussianArray2D((j, k) => dist(i, j, k), length2, length3)));
        }

        private static GammaArray1D CreateGammaArray1D(Gamma dist, int length)
        {
            return CreateGammaArray1D(i => dist, length);
        }

        private static GammaArray1D CreateGammaArray1D(Func<int, Gamma> dist, int length)
        {
            return new GammaArray1D(Util.ArrayInit(length, i => dist(i)));
        }

        private static GammaArray2D CreateGammaArray2D(Gamma dist, int length1, int length2)
        {
            return CreateGammaArray2D((i, j) => dist, length1, length2);
        }

        private static GammaArray2D CreateGammaArray2D(Func<int, int, Gamma> dist, int length1, int length2)
        {
            return new GammaArray2D(Util.ArrayInit(length1, i => CreateGammaArray1D(j => dist(i, j), length2)));
        }

        private static WishartArray1D CreateWishartArray1D(Wishart dist, int length)
        {
            return CreateWishartArray1D(i => dist, length);
        }

        private static WishartArray1D CreateWishartArray1D(Func<int, Wishart> dist, int length)
        {
            return new WishartArray1D(Util.ArrayInit(length, i => dist(i)));
        }
        
        private static WishartArray2D CreateWishartArray2D(Wishart dist, int length1, int length2)
        {
            return CreateWishartArray2D((i, j) => dist, length1, length2);
        }
        
        private static WishartArray2D CreateWishartArray2D(Func<int, int, Wishart> dist, int length1, int length2)
        {
            return new WishartArray2D(Util.ArrayInit(length1, i => CreateWishartArray1D(j => dist(i, j), length2)));
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