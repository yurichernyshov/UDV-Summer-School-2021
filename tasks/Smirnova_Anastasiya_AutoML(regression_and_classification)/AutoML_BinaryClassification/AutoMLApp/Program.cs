using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using AutoMLApp.DataStructures;
using System.Collections.Generic;

namespace AutoMLApp
{
    class Program
    {
        private const int Width = 114;

        private static readonly string BaseDatasetsRelativePath = @"../../../datasets";
        //табличка(формат .tsv) с данными для обучения модели
        private static readonly string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/wikipedia-detox-250-line-data.tsv";
        //табличка(формат .tsv) с данными для тестирования модели
        private static readonly string TestDataRelativePath = $"{BaseDatasetsRelativePath}/wikipedia-detox-250-line-test.tsv";
        //находим абсолютные пути до таблиц
        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        //папка, куда будем сохранять модель
        private static readonly string BaseModelsRelativePath = @"../../../MLModels";
        //задаем путь и имя модели, которую получим после обучения
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/AutoML_Classification.zip";
        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        //задаем время эксперимента
        private static uint ExperimentTime = 60;

        static void Main(string[] args)
        {
            //Общий контекст для всех операций ML.NET.
            //Будучи созданным пользователем, он предоставляет способ создания компонентов
            //для подготовки данных, функций, обучения, прогнозирования и оценки модели.
            var mlContext = new MLContext();

            //Создание, обучение и сохранение модели
            //возвращает натренированную модель
            BuildTrainEvaluateAndSaveModel(mlContext);

            //Создание прогноза при загрузке модели из .ZIP файла
            TestSinglePrediction(mlContext);

            //Демонстрация окончания выполнения приложения
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        //Создание, обучение и сохранение модели
        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            //используем ранее созданные директории нахождения табличек с данными и загружаем их в интерфейс
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(TrainDataPath, hasHeader: true);
            IDataView testDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(TestDataPath, hasHeader: true);

            //выводим на консоль первые 4 строчки таблицы для обучения модели
            ShowDataViewInConsole(mlContext, trainingDataView, 4);

            //создаем handler для отображения прогресса выполнения алгоритма классификации
            //на консоль будет поочередно выводится имя метрик, их прогресс и время выполнения эксперимента (обучения)
            var progressHandler = new BinaryExperimentProgressHandler();

            //начало выполнения алгоритма классификации
            ConsoleWriteHeader("=============== Running AutoML experiment ===============");
            //выводим на консоль время выполнения эксперимента
            Console.WriteLine($"Running AutoML binary classification experiment for {ExperimentTime} seconds...");

            // ExperimentResult - результат эксперимента AutoML
            // BinaryClassificationMetrics - тип метрик для эксперимента
            // создаем эксперимент с заданным временем - ExperimentTime
            // выбираем для обучение таблицу, выбранную ранее
            // указываем обработчик для отображения прогресса выполнения эксперимента
            ExperimentResult<BinaryClassificationMetrics> experimentResult = mlContext.Auto()
                .CreateBinaryClassificationExperiment(ExperimentTime)
                .Execute(trainingDataView, progressHandler: progressHandler);


            // оценка модели
            ConsoleWriteHeader("=============== Evaluating model's accuracy with test data ===============");
            // RunDetail - класс с метриками того, как обученная модель работала с данными проверки во время запуска
            RunDetail<BinaryClassificationMetrics> bestRun = experimentResult.BestRun;
            // создаем преобразователь данных
            ITransformer trainedModel = bestRun.Model;
            // используем тестовую табличку, чтобы создать прогнозы на ее основе
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions, scoreColumnName: "Score");
            PrintBinaryClassificationMetrics(bestRun.TrainerName, metrics);

            // сохраняем модель в архиве
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            return trainedModel;
        }

        // протестируем строчку с помощью созданной моделью
        private static void TestSinglePrediction(MLContext mlContext)
        {
            ConsoleWriteHeader("=============== Testing prediction ===============");
            //создаем строчку, помещаем в колонку Text
            SentimentIssue sampleStatement = new SentimentIssue { Text = "Today was the worst day ever" };

            //ищем модель по ране созданному пути 
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            //создаем прогноз 
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);
            Console.WriteLine($"=============== Created Prediction Engine OK  ===============");
            // Score
            var predictedResult = predEngine.Predict(sampleStatement);

            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(predictedResult.Prediction) ? "Toxic" : "Non Toxic")} sentiment");
            Console.WriteLine($"==================================================");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
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

        public static void PrintBinaryClassificationMetrics(string name, BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {name} binary classification model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       Area Under Curve:      {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       Area under Precision recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
            Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
            Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
            Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
            Console.WriteLine($"************************************************************");
        }

        private static void CreateRow(string message, int width)
        {
            Console.WriteLine("|" + message.PadRight(width - 2) + "|");
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

        internal static void PrintIterationException(Exception ex)
        {
            Console.WriteLine($"Exception during AutoML iteration: {ex}");
        }


        internal static void PrintBinaryClassificationMetricsHeader()
        {
            CreateRow($"{"",-4} {"Trainer",-35} {"Accuracy",9} {"AUC",8} {"AUPRC",8} {"F1-score",9} {"Duration",9}", Width);
        }

        public class BinaryExperimentProgressHandler : IProgress<RunDetail<BinaryClassificationMetrics>>
        {
            private int _iterationIndex;

            public void Report(RunDetail<BinaryClassificationMetrics> iterationResult)
            {
                if (_iterationIndex++ == 0)
                {
                    PrintBinaryClassificationMetricsHeader();
                }

                if (iterationResult.Exception != null)
                {
                    PrintIterationException(iterationResult.Exception);
                }
                else
                {
                    PrintIterationMetrics(_iterationIndex, iterationResult.TrainerName,
                        iterationResult.ValidationMetrics, iterationResult.RuntimeInSeconds);
                }
            }

        }
    }
}
