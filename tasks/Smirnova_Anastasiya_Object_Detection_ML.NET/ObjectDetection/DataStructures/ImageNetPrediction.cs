using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures
{
    public class ImageNetPrediction
    {
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
    //ImageNetPrediction является классом прогноза данных и имеет следующее поле float[]:
    //PredictedLabel содержит измерения, оценку объекта и 
    //вероятности класса для каждого ограничивающего прямоугольника,
    //обнаруженного в изображении.
}
