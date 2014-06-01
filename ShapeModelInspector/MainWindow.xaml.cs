using Microsoft.Win32;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;
using SegmentationGrid;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

using Color = System.Drawing.Color;
using Vector = MicrosoftResearch.Infer.Maths.Vector;

using Mask = MicrosoftResearch.Infer.Distributions.DistributionStructArray2D<MicrosoftResearch.Infer.Distributions.Bernoulli, bool>;
using System.ComponentModel;

namespace ShapeModelInspector
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private ShapeModel shapeModel;

        private List<Slider> traitSliders = new List<Slider>();

        private bool[,] maskToComplete;

        public MainWindow()
        {
            InitializeComponent();
        }

        #region Event handlers

        private void OnLoadShapeModelButtonClick(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openDialog = new OpenFileDialog();
            openDialog.InitialDirectory = @"C:\Users\boya\Source\BayesianShapePrior\ShapeModelLearner\bin\Release";
            if (openDialog.ShowDialog().Value)
            {
                try
                {
                    this.shapeModel = ShapeModel.Load(openDialog.FileName);
                }
                catch (Exception)
                {
                    MessageBox.Show("Something went wrong while loading a model from " + openDialog.FileName);
                    return;
                }

                this.loadMaskToCompleteButton.IsEnabled = true;

                this.traitSetupPanel.Children.Clear();
                this.traitSliders.Clear();
                for (int i = 0; i < this.shapeModel.TraitCount; ++i)
                {
                    Slider slider = new Slider();
                    slider.Value = i == this.shapeModel.TraitCount - 1 ? 1 : 0;
                    slider.Minimum = -3;
                    slider.Maximum = 3;
                    slider.TickPlacement = System.Windows.Controls.Primitives.TickPlacement.BottomRight;
                    slider.TickFrequency = 0.25;
                    slider.ValueChanged += OnSliderValueChanged;

                    traitSliders.Add(slider);
                    this.traitSetupPanel.Children.Add(slider);

                    Label sliderValueLabel = new Label();
                    sliderValueLabel.DataContext = slider;
                    sliderValueLabel.SetBinding(Label.ContentProperty, new Binding("Value"));
                    sliderValueLabel.Margin = new Thickness(5, 5, 5, 15);
                    this.traitSetupPanel.Children.Add(sliderValueLabel);
                }

                this.Sample();
            }
        }

        private void OnSliderValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            this.Sample();
        }

        private void OnLoadMaskToCompleteButtonClick(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openDialog = new OpenFileDialog();
            openDialog.InitialDirectory = @"C:\Users\boya\Source\BayesianShapePrior\Data\horses\figure_ground";
            if (openDialog.ShowDialog().Value)
            {
                try
                {
                    Bitmap bitmap = new Bitmap(openDialog.FileName);
                    this.maskToComplete = ImageHelpers.BitmapToArray(
                        ImageHelpers.CropBitmap(bitmap, this.shapeModel.GridWidth, this.shapeModel.GridHeight),
                        ImageHelpers.ColorToMaskValue);
                }
                catch (Exception)
                {
                    MessageBox.Show("Something went wrong while loading an image from " + openDialog.FileName);
                    return;
                }

                this.maskToCompleteViewer.Source = WpfImageHelpers.MaskToBitmapSource(this.maskToComplete);
                this.completeLeftMaskPartButton.IsEnabled = true;
                this.completeRightMaskPartButton.IsEnabled = true;
                this.fitModelButton.IsEnabled = true;
            }
        }

        private void OnCompleteLeftMaskPartButtonClick(object sender, RoutedEventArgs e)
        {
            var availablePixels = Util.ArrayInit(
                this.shapeModel.GridWidth,
                this.shapeModel.GridHeight,
                (i, j) => i < this.shapeModel.GridWidth / 2);
            RunCompletion(availablePixels);
        }

        private void OnCompleteRightMaskPartButtonClick(object sender, RoutedEventArgs e)
        {
            var availablePixels = Util.ArrayInit(
                this.shapeModel.GridWidth,
                this.shapeModel.GridHeight,
                (i, j) => i >= this.shapeModel.GridWidth / 2);
            RunCompletion(availablePixels);
        }

        private void OnFitModelButtonClick(object sender, RoutedEventArgs e)
        {
            this.DisableUI();

            BackgroundWorker worker = new BackgroundWorker();
            worker.DoWork += DoFitting;
            worker.RunWorkerCompleted += OnFittingFinished;
            worker.RunWorkerAsync();
        }

        #endregion

        #region Sampling

        private void Sample()
        {
            //Rand.Restart(666);

            double[] traits = Util.ArrayInit(this.shapeModel.TraitCount, i => this.traitSliders[i].Value);
            ShapeModelSample sample = this.shapeModel.Sample(new[] { traits }, false, true)[0];
            this.maskViewer.Source = WpfImageHelpers.MaskToBitmapSource(sample.Labels);
            this.shapeViewer.Source = WpfImageHelpers.BitmapToBitmapSource(DrawSample(sample));
        }

        private static Bitmap DrawSample(ShapeModelSample sample)
        {
            return ImageHelpers.DrawShape(
                sample.ShapeModel.GridWidth,
                sample.ShapeModel.GridHeight,
                sample.ShapePartLocations,
                sample.ShapePartOrientations);
        }

        #endregion

        #region Completion

        private void RunCompletion(bool[,] availablePixels)
        {
            this.DisableUI();
            
            BackgroundWorker worker = new BackgroundWorker();
            worker.DoWork += DoCompletion;
            worker.RunWorkerCompleted += OnCompletionFinished;
            worker.RunWorkerAsync(availablePixels);
        }

        private void OnCompletionFinished(object sender, RunWorkerCompletedEventArgs e)
        {
            this.EnableUI();

            this.completedMaskViewer.Source = WpfImageHelpers.MaskToBitmapSource((bool[,])e.Result);
        }

        private void DoCompletion(object sender, DoWorkEventArgs e)
        {
            ShapeModelInference inference = new ShapeModelInference();
            inference.SetBeliefs(this.shapeModel);
            inference.MaskCompletionIterationCount = 300;
            inference.IterationsBetweenCallbacks = 5;
            inference.MaskCompletionProgress += OnMaskCompletionProgressChange;

            Mask completedMask = inference.CompleteMasks(new[] { this.maskToComplete }, new[] { (bool[,])e.Argument })[0];
            e.Result = GetCertainCompletedMask(completedMask);
        }

        private void OnMaskCompletionProgressChange(object sender, MaskCompletionProgressEventArgs e)
        {
            ShapeModelInference inferenceEngine = (ShapeModelInference)sender;
            double progress = 100.0 * e.IterationsCompleted / inferenceEngine.MaskCompletionIterationCount;
            this.Dispatcher.Invoke(
                () =>
                {
                    this.completionProgressBar.Value = progress;
                    this.completedMaskViewer.Source = WpfImageHelpers.MaskToBitmapSource(GetCertainCompletedMask(e.CompletedMasks[0]));
                });
        }

        private bool[,] GetCertainCompletedMask(Mask mask)
        {
            return Util.ArrayInit(this.shapeModel.GridWidth, this.shapeModel.GridHeight, (i, j) => mask[i, j].GetMode());
        }

        #endregion

        #region Fitting

        private void DoFitting(object sender, DoWorkEventArgs e)
        {
            ShapeModelInference inference = new ShapeModelInference();
            inference.SetBeliefs(this.shapeModel);
            inference.TrainingIterationCount = 1000;
            inference.IterationsBetweenCallbacks = 5;
            inference.ShapeModelLearningProgress += OnShapeFittingProgressChange;

            e.Result = inference.LearnModel(new[] { this.maskToComplete }, false);
        }

        private void OnFittingFinished(object sender, RunWorkerCompletedEventArgs e)
        {
            this.EnableUI();

            Tuple<ShapeModel, ShapeFittingInfo> modelFittingInfo = (Tuple<ShapeModel, ShapeFittingInfo>)e.Result;
            this.completedMaskViewer.Source = WpfImageHelpers.BitmapToBitmapSource(GetFittingMask(modelFittingInfo.Item1, modelFittingInfo.Item2));
        }

        private void OnShapeFittingProgressChange(object sender, ShapeModelLearningProgressEventArgs e)
        {
            ShapeModelInference inferenceEngine = (ShapeModelInference)sender;
            double progress = 100.0 * e.IterationsCompleted / inferenceEngine.MaskCompletionIterationCount;
            this.Dispatcher.Invoke(
                () =>
                {
                    this.completionProgressBar.Value = progress;
                    this.completedMaskViewer.Source = WpfImageHelpers.BitmapToBitmapSource(GetFittingMask(e.ShapeModel, e.FittingInfo));
                });
        }

        private Bitmap GetFittingMask(ShapeModel shapeModel, ShapeFittingInfo fittingInfo)
        {
            Vector[] meanLocations = Util.ArrayInit(fittingInfo.ShapePartLocations[0].Count, i => Vector.FromArray(Util.ArrayInit(2, j => fittingInfo.ShapePartLocations[0][i][j].GetMean())));
            PositiveDefiniteMatrix[] meanOrientations = Util.ArrayInit(fittingInfo.ShapePartOrientations[0].Count, j => fittingInfo.ShapePartOrientations[0][j].GetMean());
            return ImageHelpers.DrawShape(shapeModel.GridWidth, shapeModel.GridHeight, meanLocations, meanOrientations);
        }

        #endregion

        #region UI helpers
        
        private void DisableUI()
        {
            this.loadShapeModelButton.IsEnabled = false;
            this.loadMaskToCompleteButton.IsEnabled = false;
            this.completeLeftMaskPartButton.IsEnabled = false;
            this.completeRightMaskPartButton.IsEnabled = false;
            this.fitModelButton.IsEnabled = false;
        }

        private void EnableUI()
        {
            this.loadShapeModelButton.IsEnabled = true;
            this.loadMaskToCompleteButton.IsEnabled = true;
            this.completeLeftMaskPartButton.IsEnabled = true;
            this.completeRightMaskPartButton.IsEnabled = true;
            this.fitModelButton.IsEnabled = true;
        }

        #endregion
    }
}
