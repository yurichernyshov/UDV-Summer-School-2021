using Microsoft.AspNetCore.Mvc;
using CoreApi.ML.Models;
using Microsoft.Extensions.ML;

namespace CoreApi.Controllers
{
    public class ViewController : Controller
    {

        private readonly PredictionEnginePool<SentimentData, SentimentPrediction> predictionEnginePool;

        public ViewController(PredictionEnginePool<SentimentData, SentimentPrediction> predictionEnginePool)
        {
            this.predictionEnginePool = predictionEnginePool;
        }

        [HttpGet("")]
        public ActionResult Index()
        {
            //ViewBag.Message = null;
            return View("Views/PredictPage.cshtml");
        }
        
        [HttpPost("Predict")]
        public ActionResult UpdateIndex([FromForm] SentimentData comment)
        {
            SentimentPrediction prediction = predictionEnginePool.Predict(modelName: "SentimentAnalysisModel", example: comment);
            comment.Sentiment = prediction.Prediction;
            ViewBag.Message = prediction;
            return Index();
        }
    }
}
