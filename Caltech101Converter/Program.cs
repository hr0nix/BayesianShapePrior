using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Caltech101Converter
{
    class Program
    {
        static void Main(string[] args)
        {
            const string Dataset = "Motorbikes_16";
            const string DataDir = "../../../Data/Caltech101/Annotations";
            const string OutputDir = "../../../Data/Caltech101/AnnotationsConverted";
            const int SamplingRate = 15;

            string outputDirFull = Path.Combine(OutputDir, Dataset);
            string subsetOutputDirFull = Path.Combine(outputDirFull, "Subset");
            if (Directory.Exists(outputDirFull))
            {
                Directory.Delete(outputDirFull, true);
            }
            
            Directory.CreateDirectory(outputDirFull);
            Directory.CreateDirectory(subsetOutputDirFull);
            
            int imageIndex = 0;
            foreach (string annotationFileNameFull in Directory.EnumerateFiles(Path.Combine(DataDir, Dataset), "*.mat"))
            {
                var data = MatlabReader.Read(annotationFileNameFull);
                var contour = (Matrix)data["obj_contour"];

                using (Bitmap bitmap = new Bitmap(200, 150))
                using (Graphics graphics = Graphics.FromImage(bitmap))
                {
                    Point[] path = Util.ArrayInit(contour.Cols, i => new Point(Convert.ToInt32(contour[0, i]), Convert.ToInt32(contour[1, i])));
                    graphics.Clear(Color.Black);
                    graphics.FillClosedCurve(Brushes.White, path);

                    string bitmapFileName = Path.GetFileNameWithoutExtension(annotationFileNameFull) + ".png";
                    bitmap.Save(Path.Combine(outputDirFull, bitmapFileName));

                    if (imageIndex++ % SamplingRate == 0)
                    {
                        bitmap.Save(Path.Combine(subsetOutputDirFull, bitmapFileName));
                    }
                }
            }
        }
    }
}
