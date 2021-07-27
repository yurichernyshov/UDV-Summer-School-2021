using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis
{
	class Program
	{
		// поле для хранения пути к файлу загруженного набора данных
		static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "comments.txt");
		static void Main(string[] args)
		{
			MLContext mlContext = new MLContext();
			TrainTestData splitDataView = LoadData(mlContext);

			ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

			Evaluate(mlContext, model, splitDataView.TestSet);
			UseModelWithSingleItem(mlContext, model);
			UseModelWithBatchItems(mlContext, model);
		}


		// метод ниже извлекает и преобразует данные, обучает модель, делает прогноз на основе тестовых данных и возвращает модель
		public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
		{
			// FeaturizeText() преобразует текстовый столбец в числовой тип столбца Features и добавляет его в новый столбец набора данных
			var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
			// задача машинного обучения к определениям преобразования данных
				.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

			// возвращение обученной модели
			Console.WriteLine("=============== Create and Train the Model ===============");
			// Fit() обучает модель, преобразуя набор данных и применяя обучение
			var model = estimator.Fit(splitTrainSet);
			Console.WriteLine("=============== End of training ===============");
			Console.WriteLine();

			return model;
		}


		// метод ниже использует тестовые данные для проверки после обучения модели:
		// загружает тестовый набор данных, создаёт оценщик BinaryClassification, оценивает модель, создаёт и отображает метрики
		public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
		{
			Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");

			// Transform() используется для прогнозирования нескольких предоставленных входных строк тестового набора данных
			IDataView predictions = model.Transform(splitTestSet);
			CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

			Console.WriteLine();
			Console.WriteLine("Model quality metrics evaluation");
			Console.WriteLine("--------------------------------");
			// Accuracy получает точность модели, которая представляет собой долю правильных прогнозов в тестовом наборе
			Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
			// AreaUnderRocCurve показывает, насколько уверенно модель классифицирует положительные и отрицательные классы
			Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
			// мера баланса между точностью и отзывом
			Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
			Console.WriteLine("=============== End of model evaluation ===============");

		}
		// создаёт одиночный отзыв тестовых данных,
		// прогнозирует настроение на основе тестовых данных,
		// объединяет тестовые данные и прогнозы для создания отчетов,
		// отображает прогнозируемые результаты
		private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
		{
			PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
			
			// для проверки предсказания обученной модели
			SentimentData sampleStatement = new SentimentData
			{
				SentimentText = "This was a very bad steak"
			};

			// Predict() делает прогноз для одной строки данных
			var resultPrediction = predictionFunction.Predict(sampleStatement);

			Console.WriteLine();
			Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

			Console.WriteLine();
			Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

			Console.WriteLine("=============== End of Predictions ===============");
			Console.WriteLine();
		}

		// создает данные пакетного тестирования,
		// прогнозирует настроение на основе тестовых данных,
		// объединяет тестовые данные и прогнозы для создания отчетов,
		// отображает прогнозируемые результаты
		public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
		{
			// для проверки предсказания обученной модели
			IEnumerable<SentimentData> sentiments = new[]
			{
				new SentimentData
				{
					SentimentText = "This was a horrible meal"
				},
				new SentimentData
				{
					SentimentText = "I love this spaghetti."
				}
			};

			IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

			IDataView predictions = model.Transform(batchComments);

			// используем модель, чтобы предсказать, являются ли данные отзывы положительными (1) или отрицательными (0)
			IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

			Console.WriteLine();

			Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

			foreach (SentimentPrediction prediction in predictedResults)
			{
				Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
			}
			Console.WriteLine("=============== End of predictions ===============");
		}

		// метод LoadData() загружает данные, разбивает данные на обучающие и тестовые наборы данных
		// и возвращает разделенные наборы данных для поездов и тестов

		public static TrainTestData LoadData(MLContext mlContext)
		{
			// LoadFromTextFile() определяет схему данных и считывает файл, принимает переменные пути к данным и возвращает IDataView
			IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

			// разделяет загруженные данные на необходимые наборы данных
			// процент данных набора тестов указывается с помощью параметра testFraction
			TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
			return splitDataView;
		}
	}


}
