using Microsoft.ML.Data;
using System;

namespace CoreApi.ML.Models
{
    public class SentimentPrediction: SentimentData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }

        public override string ToString()
        {
            var sentiment = Prediction ? "Positive" : "Negative";

            return String.Format("Комментарий: {0} \nЭмоциональный окрас: {1}", SentimentText, sentiment);
        }
    }
}
