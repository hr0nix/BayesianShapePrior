using MicrosoftResearch.Infer.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ShapeModelInspector
{
    [StructLayout(LayoutKind.Sequential)]
    struct PixelColor
    {
        public byte Blue;
        public byte Green;
        public byte Red;
        public byte Alpha;

        public PixelColor(byte red, byte green, byte blue, byte alpha)
        {
            this.Red = red;
            this.Green = green;
            this.Blue = blue;
            this.Alpha = alpha;
        }

        public static PixelColor FromColor(System.Drawing.Color color)
        {
            return new PixelColor(color.R, color.G, color.B, 255);
        }
    }

    static class WpfImageHelpers
    {
        public const double Dpi = 96;

        public static PixelColor[,] GetPixelsFromBitmapSource(BitmapSource source)
        {
            if (source.Format != PixelFormats.Bgra32)
                source = new FormatConvertedBitmap(source, PixelFormats.Bgra32, null, 0);

            int width = source.PixelWidth;
            int height = source.PixelHeight;
            byte[] pixelBytes = new byte[height * width * 4];
            source.CopyPixels(pixelBytes, width * 4, 0);

            PixelColor[,] pixels = new PixelColor[width, height];
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    pixels[x, y] = new PixelColor
                    {
                        Blue = pixelBytes[(y * width + x) * 4 + 0],
                        Green = pixelBytes[(y * width + x) * 4 + 1],
                        Red = pixelBytes[(y * width + x) * 4 + 2],
                        Alpha = pixelBytes[(y * width + x) * 4 + 3],
                    };

            return pixels;
        }

        public static WriteableBitmap PixelsToBitmapSource(PixelColor[,] pixels)
        {
            int width = pixels.GetLength(0);
            int height = pixels.GetLength(1);
            byte[] pixelData = new byte[width * height * 4];
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                {
                    PixelColor color = pixels[x, y];
                    pixelData[(y * width + x) * 4 + 0] = color.Blue;
                    pixelData[(y * width + x) * 4 + 1] = color.Green;
                    pixelData[(y * width + x) * 4 + 2] = color.Red;
                    pixelData[(y * width + x) * 4 + 3] = color.Alpha;
                }

            WriteableBitmap bitmap = new WriteableBitmap(width, height, Dpi, Dpi, PixelFormats.Bgra32, null);
            bitmap.WritePixels(new Int32Rect(0, 0, width, height), pixelData, bitmap.PixelWidth * 4, 0);
            return bitmap;
        }

        public static WriteableBitmap BitmapToBitmapSource(Bitmap bitmap)
        {
            PixelColor[,] pixels = Util.ArrayInit(bitmap.Width, bitmap.Height, (i, j) => PixelColor.FromColor(bitmap.GetPixel(i, j)));
            return PixelsToBitmapSource(pixels);
            
        }

        public static BitmapSource MaskToBitmapSource(bool[,] mask)
        {
            PixelColor[,] pixels = new PixelColor[mask.GetLength(0), mask.GetLength(1)];
            PixelColor objectColor = new PixelColor(255, 0, 0, 120);
            PixelColor backgroundColor = new PixelColor(0, 0, 0, 0);
            for (int i = 0; i < mask.GetLength(0); ++i)
                for (int j = 0; j < mask.GetLength(1); ++j)
                    pixels[i, j] = mask[i, j] ? objectColor : backgroundColor;

            return PixelsToBitmapSource(pixels);
        }

        public static BitmapSource ResizeImage(BitmapSource source, int width, int height)
        {
            Rect rect = new Rect(0, 0, width, height);
            DrawingVisual drawingVisual = new DrawingVisual();
            using (DrawingContext drawingContext = drawingVisual.RenderOpen())
                drawingContext.DrawImage(source, rect);

            RenderTargetBitmap resizedImage = new RenderTargetBitmap(
                (int)rect.Width, (int)rect.Height,
                Dpi, Dpi,
                PixelFormats.Default);
            resizedImage.Render(drawingVisual);

            return resizedImage;
        }

        public static BitmapSource FixDpi(BitmapSource image)
        {
            return ResizeImage(image, image.PixelWidth, image.PixelHeight);
        }
    }
}
