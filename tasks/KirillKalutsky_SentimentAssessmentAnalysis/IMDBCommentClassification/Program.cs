using IMDBCommentClassification.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using static Microsoft.ML.DataOperationsCatalog;

namespace IMDBCommentClassification
{
    static class Program
    {
        static readonly string dataPath = Path.Combine("..","..","..","Data","allData.txt");

        static readonly string modelPath = Path.Combine("..","..","..","Data","model.zip");
       
        private static Dictionary<string, int> dirWithData =
            new Dictionary<string, int>()
            {
                { Path.Combine("..","..","..","Data","Datasets","Test","neg"), 0 },
                { Path.Combine("..","..","..","Data","Datasets","Test","pos"), 1 },
                { Path.Combine("..","..","..","Data","Datasets","Train","neg"), 0},
                { Path.Combine("..","..","..","Data","Datasets","Train","pos"), 1 }
            };
       
        static void Main(string[] args)
        {
            if(!File.Exists(dataPath))
                JoinFiles(dirWithData,dataPath); 
            
            WorkWithMLModel();
        }

        private static void WorkWithMLModel()
        {
            var mlContext = new MLContext();

            ITransformer model;

            try
            {
                DataViewSchema modelSchema;
                model = mlContext.Model.Load(modelPath, out modelSchema);
            }
            catch (Exception e)
            {
                var dataView = mlContext.Data.ShuffleRows(mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: false));

                var splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

                model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

                Evaluate(mlContext, model, splitDataView.TestSet);

                mlContext.Model.Save(model, dataView.Schema, modelPath);
            }

            var listWords = new List<string>()
            {
                "brilliant",
                "fantastic",
                "amazing",
                "good",
                "bad",
                "awful",
                "crap",
                "terrible",
                "trash"
            };

            foreach (var word in listWords)
            {
                UseModel(mlContext, model, word);
            }

            Console.WriteLine("Write your sentiment, please!");
            while (true)
            {
                var input = Console.ReadLine();
                if (input.Equals("stop")) 
                    break;
                UseModel(mlContext, model, input);
            }
        }

        private static void JoinFiles(Dictionary<string,int> directories,string outPath)
        {
            using (StreamWriter w = new StreamWriter(outPath, false, Encoding.UTF8))
            {
                foreach(var e in directories)
                    WriteInFile(e.Key, w, e.Value);
            }
        }

        private static void WriteInFile(string dirPath, StreamWriter resultFile, int booleanValue)
        {
            Regex rgx = new Regex("[^a-zA-Z ]");

            foreach (var file in Directory.GetFiles(dirPath))
            {
                var str = rgx.Replace(File.ReadAllLines(file)[0], "").ToLower();
                resultFile.WriteLine
                    (
                        str + '\t' + booleanValue
                    );
            }
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            //Метод FeaturizeText(), преобразует текстовый столбец (SentimentText) в числовой столбец типа ключа Features,
            //который переводит слова в числовое представлени для использования алгоритмом машинного обучения

            //Используем бинарную классификацию, так как нам нужно определить является высказывание положительныи или отрицательным
            var estimator = 
                mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static void UseModel(MLContext mlContext, ITransformer model, string comment)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = comment
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }
    }
}
