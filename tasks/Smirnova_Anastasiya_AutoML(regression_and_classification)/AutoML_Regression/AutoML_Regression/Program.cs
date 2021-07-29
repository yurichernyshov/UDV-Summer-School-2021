using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using PLplot;
using AutoML_Regression.DataStructures;

namespace AutoML_Regression
{
    internal static class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string BaseDatasetsRelativePath = @"../../../datasets";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/taxi-fare-train.csv";
        private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}/taxi-fare-test.csv";
        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static string BaseModelsRelativePath = @"../../../ML_Model";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/TaxiFareModel.zip";
        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        private static string LabelColumnName = "FareAmount";
        private static uint ExperimentTime = 60;
        private const int Width = 114;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // Создание, обучение и сохранение модели 
            BuildTrainEvaluateAndSaveModel(mlContext);

            // Проверка на единичном тесте модели, сохраненной в папке
            TestSinglePrediction(mlContext);


            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();
        }

        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TrainDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TestDataPath, hasHeader: true, separatorChar: ',');

            ShowDataViewInConsole(mlContext, trainingDataView, 4);
            
            var progressHandler = new RegressionExperimentProgressHandler();

            ConsoleWriteHeader("=============== Training the model ===============");
            Console.WriteLine($"Running AutoML regression experiment for {ExperimentTime} seconds...");
            ExperimentResult<RegressionMetrics> experimentResult = mlContext.Auto()
                .CreateRegressionExperiment(ExperimentTime)
                .Execute(trainingDataView, LabelColumnName, progressHandler: progressHandler);

            
            ConsoleWriteHeader("===== Evaluating model's accuracy with test data =====");
            RunDetail<RegressionMetrics> best = experimentResult.BestRun;
            ITransformer trainedModel = best.Model;
            IDataView predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: LabelColumnName, scoreColumnName: "Score");
            
            PrintRegressionMetrics(best.TrainerName, metrics);

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            return trainedModel;
        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            ConsoleWriteHeader("=============== Testing prediction engine ===============");

            // Sample: 
            // vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            // VTS,1,1,1140,3.75,CRD,15.5

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(trainedModel);

            // Score
            var predictedResult = predEngine.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {predictedResult.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }


        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new string('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ShowDataViewInConsole(MLContext mlContext, IDataView dataView, int numberOfRows)
        {
            string msg = string.Format("Show data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            var preViewTransformedData = dataView.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                Console.WriteLine(lineToPrint + "\n");
            }
        }

        public static void PrintRegressionMetrics(string name, RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name} regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        public class RegressionExperimentProgressHandler : IProgress<RunDetail<RegressionMetrics>>
        {
            private int _iterationIndex;

            public void Report(RunDetail<RegressionMetrics> iterationResult)
            {
                if (_iterationIndex++ == 0)
                {
                    CreateRow($"{"",-4} {"Trainer",-35} {"RSquared",8} {"Absolute-loss",13} {"Squared-loss",12} {"RMS-loss",8} {"Duration",9}", Width);
                }

                if (iterationResult.Exception != null)
                {
                    Console.WriteLine($"Exception during AutoML iteration: {iterationResult.Exception}");
                }
                else
                {
                    PrintIterationMetrics(_iterationIndex, iterationResult.TrainerName,
                        iterationResult.ValidationMetrics, iterationResult.RuntimeInSeconds);
                }
            }
        }

        internal static void PrintIterationMetrics(int iteration, string trainerName, BinaryClassificationMetrics metrics, double? runtimeInSeconds)
        {
            CreateRow($"{iteration,-4} {trainerName,-35} {metrics?.Accuracy ?? double.NaN,9:F4} {metrics?.AreaUnderRocCurve ?? double.NaN,8:F4} {metrics?.AreaUnderPrecisionRecallCurve ?? double.NaN,8:F4} {metrics?.F1Score ?? double.NaN,9:F4} {runtimeInSeconds.Value,9:F1}", Width);
        }

        internal static void PrintIterationMetrics(int iteration, string trainerName, MulticlassClassificationMetrics metrics, double? runtimeInSeconds)
        {
            CreateRow($"{iteration,-4} {trainerName,-35} {metrics?.MicroAccuracy ?? double.NaN,14:F4} {metrics?.MacroAccuracy ?? double.NaN,14:F4} {runtimeInSeconds.Value,9:F1}", Width);
        }

        internal static void PrintIterationMetrics(int iteration, string trainerName, RegressionMetrics metrics, double? runtimeInSeconds)
        {
            CreateRow($"{iteration,-4} {trainerName,-35} {metrics?.RSquared ?? double.NaN,8:F4} {metrics?.MeanAbsoluteError ?? double.NaN,13:F2} {metrics?.MeanSquaredError ?? double.NaN,12:F2} {metrics?.RootMeanSquaredError ?? double.NaN,8:F2} {runtimeInSeconds.Value,9:F1}", Width);
        }

        internal static void PrintIterationMetrics(int iteration, string trainerName, RankingMetrics metrics, double? runtimeInSeconds)
        {
            CreateRow($"{iteration,-4} {trainerName,-15} {metrics?.NormalizedDiscountedCumulativeGains[0] ?? double.NaN,9:F4} {metrics?.NormalizedDiscountedCumulativeGains[2] ?? double.NaN,9:F4} {metrics?.NormalizedDiscountedCumulativeGains[9] ?? double.NaN,9:F4} {metrics?.DiscountedCumulativeGains[9] ?? double.NaN,9:F4} {runtimeInSeconds.Value,9:F1}", Width);
        }

        private static void CreateRow(string message, int width)
        {
            Console.WriteLine("|" + message.PadRight(width - 2) + "|");
        }
    }
}
