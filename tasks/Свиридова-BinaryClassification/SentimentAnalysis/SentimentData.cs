using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    // класс входного набора данных
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    // класс прогнозирования, используемый после обучения модели
    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
