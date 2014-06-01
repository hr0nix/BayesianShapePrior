using System;
using System.Drawing;
using MicrosoftResearch.Infer.Maths;
using System.Diagnostics;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.IO;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer.Distributions;

namespace SegmentationGrid
{
    public static class ImageHelpers
    {
        public static Bitmap ArrayToBitmap<T>(T[,] array, Func<T, Color> converter)
        {
            int width = array.GetLength(0);
            int height = array.GetLength(1);
            var result = new Bitmap(width, height);
            for (int i = 0; i < width; ++i)
                for (int j = 0; j < height; ++j)
                    result.SetPixel(i, j, converter(array[i, j]));
            return result;
        }

        public static T[,] BitmapToArray<T>(Bitmap bitmap, Func<Color, T> converter)
        {
            var result = new T[bitmap.Width, bitmap.Height];
            for (int i = 0; i < bitmap.Width; ++i)
                for (int j = 0; j < bitmap.Height; ++j)
                    result[i, j] = converter(bitmap.GetPixel(i, j));
            return result;
        }

        public static Bitmap DownscaleBitmap(Bitmap bitmap, int maxNewWidth, int maxNewHeight)
        {
            int currentWidth = bitmap.Width;
            int currentHeight = bitmap.Height;
            double widthScale = Math.Min((double)maxNewWidth / currentWidth, 1);
            double heightScale = Math.Min((double)maxNewHeight / currentHeight, 1);
            double scale = Math.Min(widthScale, heightScale);
            return new Bitmap(bitmap, (int)Math.Round(currentWidth * scale), (int)Math.Round(currentHeight * scale));
        }

        public static Bitmap LoadScaledImage(string fileName, int maxWidth, int maxHeight)
        {
            var bitmap = new Bitmap(fileName);
            return DownscaleBitmap(bitmap, maxWidth, maxHeight);
        }

        public static Vector[,] LoadImageAsColors(string fileName)
        {
            var bitmap = new Bitmap(fileName);
            return BitmapToArray(bitmap, c => Vector.FromArray(c.R / 255.0, c.G / 255.0, c.B / 255.0));
        }

        public static Bitmap CropBitmap(Bitmap bitmap, int newWidth, int newHeight)
        {
            Debug.Assert(newWidth <= bitmap.Width && newHeight <= bitmap.Height);

            Rectangle subImageRect = new Rectangle((bitmap.Width - newWidth) / 2, (bitmap.Height - newHeight) / 2, newWidth, newHeight);
            return bitmap.Clone(subImageRect, bitmap.PixelFormat);
        }

        public static bool[][,] LoadImagesAsSameSizeMasks(string path, int maxImagesToLoad, int maxImagesToDetermineSize, int imagesToSkip)
        {
            Debug.Assert(maxImagesToDetermineSize >= maxImagesToLoad + imagesToSkip);
            
            List<Bitmap> bitmaps = new List<Bitmap>();

            int minWidth = int.MaxValue, minHeight = int.MaxValue;
            int imagesSeen = 0;
            foreach (string imagePath in Directory.EnumerateFiles(path))
            {
                if (imagesSeen++ >= maxImagesToDetermineSize)
                {
                    break;
                }
                
                // Determine size based on all images in the folder
                Bitmap bitmap = new Bitmap(imagePath);
                minWidth = Math.Min(minWidth, bitmap.Width);
                minHeight = Math.Min(minHeight, bitmap.Height);

                if (imagesToSkip-- <= 0 && bitmaps.Count < maxImagesToLoad)
                {
                    bitmaps.Add(bitmap);
                }
            }

            bool[][,] result = new bool[bitmaps.Count][,];
            for (int i = 0; i < bitmaps.Count; ++i)
            {
                Bitmap croppedBitmap = CropBitmap(bitmaps[i], minWidth, minHeight);
                result[i] = ImageHelpers.BitmapToArray(croppedBitmap, ColorToMaskValue);
            }

            return result;
        }

        public static Color DoubleToColor(double value, double minValue, double maxValue)
        {
            value = Math.Max(minValue, Math.Min(maxValue, value));
            double averagePenalty = 0.5 * (maxValue + minValue);
            double relativeDeviation = 2 * (value - averagePenalty) / (maxValue - minValue);
            Debug.Assert(relativeDeviation >= -1 && relativeDeviation <= 1);
            return Color.FromArgb((int)(Math.Max(relativeDeviation, 0) * 255), 0, (int)(Math.Max(-relativeDeviation, 0) * 255));
        }

        public static bool ColorToMaskValue(Color color)
        {
            const int threshold = 10; // To work around jpeg compression artifacts
            return color.R > threshold || color.G > threshold || color.B > threshold;
        }

        public static void DrawShapeEllipse(int gridWidth, int gridHeight, Graphics graphics, Pen pen, Vector pos, PositiveDefiniteMatrix ellipsePrecisionMatrix)
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
                    graphics.DrawLine(pen, gridWidth * (float)prevPoint[0], gridHeight * (float)prevPoint[1], gridWidth * (float)point[0], gridHeight * (float)point[1]);
                }

                prevPoint = point;
            }
        }

        public static Bitmap DrawShape(int gridWidth, int gridHeight, Vector[] shapePartLocations, PositiveDefiniteMatrix[] shapePartOrientations)
        {
            double[,] mask = Util.ArrayInit(
                gridWidth,
                gridHeight,
                (x, y) =>
                {
                    double prob = 1.0;
                    for (int i = 0; i < shapePartLocations.Length; ++i)
                    {
                        Bernoulli partLabelDistr = LabelFromShapeFactorizedOps.LabelAverageConditional(
                            Vector.FromArray((x + 0.5) / gridWidth, (y + 0.5) / gridHeight),
                            shapePartLocations[i][0],
                            shapePartLocations[i][1],
                            shapePartOrientations[i]);
                        prob *= (1 - partLabelDistr.GetProbTrue());
                    }

                    return 1 - prob;
                });

            Bitmap result = ImageHelpers.ArrayToBitmap(mask, p => PenaltyToColor(p, 0, 1));

            using (Graphics graphics = Graphics.FromImage(result))
            {
                int index = 1;
                for (int i = 0; i < shapePartLocations.Length; ++i)
                {
                    Color color = Color.FromArgb(255 * (index & 1), 255 * ((index >> 1) & 1), 255 * ((index >> 2) & 1));
                    Pen pen = new Pen(color);
                    DrawShapeEllipse(gridWidth, gridHeight, graphics, pen, Vector.FromArray(shapePartLocations[i][0], shapePartLocations[i][1]), shapePartOrientations[i]);
                    ++index;
                }
            }

            return result;
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
