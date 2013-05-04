using System;
using System.Drawing;
using MicrosoftResearch.Infer.Maths;

namespace SegmentationGrid
{
    internal static class ImageHelpers
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

        public static Vector[,] LoadImageAsColors(string fileName, int maxWidth, int maxHeight)
        {
            var bitmap = LoadScaledImage(fileName, maxWidth, maxHeight);
            return BitmapToArray(bitmap, c => Vector.FromArray(c.R / 255.0, c.G / 255.0, c.B / 255.0));
        }

        public static int[,] LoadImageAsMask(string fileName, int maxWidth, int maxHeight)
        {
            return LoadImageAsMask(fileName, maxWidth, maxHeight, c => c.R > 128 ? 1 : 0);
        }

        public static int[,] LoadImageAsMask(string fileName, int maxWidth, int maxHeight, Func<Color, int> conversionFunc)
        {
            var bitmap = LoadScaledImage(fileName, maxWidth, maxHeight);
            return BitmapToArray(bitmap, conversionFunc);
        }

        
    }
}
