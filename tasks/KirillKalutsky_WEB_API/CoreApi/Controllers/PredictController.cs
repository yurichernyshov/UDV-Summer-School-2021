using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using System.Diagnostics;
using CoreApi.ML.Models;

namespace CoreApi.Controllers
{
    [Route("")]
    [ApiController]
    public class PredictController : ControllerBase
    {
        private readonly PredictionEnginePool<SentimentData, SentimentPrediction> predictionEnginePool;

        public PredictController(PredictionEnginePool<SentimentData, SentimentPrediction> predictionEnginePool)
        {
            this.predictionEnginePool = predictionEnginePool;
        }

        [HttpPost("")]
        public string PredictFromForm([FromBody]SentimentData comment)
        {
            SentimentPrediction prediction = predictionEnginePool.Predict(modelName: "SentimentAnalysisModel", example: comment);

            Debug.WriteLine(prediction);

            return prediction.ToString();
        }
        
    }
}
