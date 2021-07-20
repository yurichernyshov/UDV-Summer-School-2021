using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures
{
    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
        {
            return Directory
                .GetFiles(imageFolder)
                .Where(filePath => Path.GetExtension(filePath) != ".md")
                .Select(filePath => new ImageNetData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
        }
    }
    //ImageNetData содержит метод ReadFromFile, 
    //    который загружает несколько файлов изображений, 
    //    хранящихся по указанном пути imageFolder, и возвращает их 
    //    в виде коллекции объектов ImageNetData.
}
