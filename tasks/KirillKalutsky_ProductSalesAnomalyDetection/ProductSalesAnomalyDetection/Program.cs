using Microsoft.ML;
using Microsoft.ML.Data;
using ProductSalesAnomalyDetection.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.Transforms.TimeSeries;

namespace ProductSalesAnomalyDetection
{
    class Program
    {
        public static readonly string dataPath = Path.Combine("..", "..", "..", "Data", "product-sales.txt");

        private static readonly int docsize = File.ReadAllLines(dataPath).Length - 1;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            DetectSpike(mlContext, docsize);

            DetectChangepoint(mlContext, docsize);
        }

        static void DetectSpike(MLContext mlContext, int docSize)
        {
            Console.WriteLine("Detect temporary changes in pattern");
            //Используем IidSpikeEstimator, чтобы обучить модель для обнаружения пиковых значений
            //Прогнозирует пики в независимом равномерно распределенном ряде времени на основе адаптивных оценок плотности ядра и мартингаланых оценок
            IidSpikeEstimator iidSpikeEstimator = mlContext.Transforms.DetectIidSpike 
                (
                    outputColumnName: nameof(ProductSalesPrediction.Prediction),
                    inputColumnName: nameof(ProductSalesData.ProductSales),
                    confidence: 95.0,
                    pvalueHistoryLength: docSize / 4
                );

            //Данные столбца являются вектором Double. Вектор содержит 3 элемента: Alert, Необработанный показатель и p

            //В результатах обнаружения пиковых значений будут отображены следующие сведения:
            //Alert указывает на оповещение о пиковом значении для заданной точки данных
            //Score является значением ProductSales для заданной точки данных в наборе данных
            //P - Value — "P" означает вероятность. Чем ближе p - значение к 0, тем больше вероятность того, что точка данных аномальна
            Console.WriteLine("Alert\tScore\tP-Value");
            FindAnomalies<IidSpikeEstimator, IidSpikeDetector>(iidSpikeEstimator, mlContext, "Spike detected");
        }

        static void DetectChangepoint(MLContext mlContext, int docSize)
        {
            Console.WriteLine("Detect persistent changes in pattern");
            //Используем IidChangePointEstimator, чтобы обучить модель для обнаружения точек изменения
            //Прогнозирует точки изменения в независимом равномерно распределенном временном ряде на основе адаптивных оценок плотности ядра и мартингаланых оценок
            IidChangePointEstimator iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint 
                (
                    outputColumnName: nameof(ProductSalesPrediction.Prediction),
                    inputColumnName: nameof(ProductSalesData.ProductSales),
                    confidence: 95.0,
                    changeHistoryLength: docSize / 4
                );

            //Данные столбца являются вектором Double . Вектор содержит 4 элемента: Alert, Необработанный показатель, p и мартингала оценку.
           
            //Alert указывает на оповещение о точке изменений для заданной точки данных
            //Score является значением ProductSales для заданной точки данных в наборе данных
            //P-Value — "P" означает вероятность. Чем ближе p-значение к 0, тем больше вероятность того, что точка данных аномальна
            //Martingale value — используется для определения того, насколько "странной" является точка данных, на основе последовательности P-значений
            Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");
            FindAnomalies<IidChangePointEstimator, IidChangePointDetector>(iidChangePointEstimator, mlContext, "Alert is on, predicted changepoint");
        }

        static void FindAnomalies<T, R>(T anomalyPointEstimator, MLContext context, string message) 
            where R: IidAnomalyDetectionBaseWrapper
            where T : TrivialEstimator<R>
        {
            //Cоздаеv пустой объект представления данных с правильной схемой для использования в качестве входных данных для метода Fit()
            var emptyDataView = context.Data.LoadFromEnumerable(new List<ProductSalesData>());

            var productSales = context.Data.LoadFromTextFile<ProductSalesData>(path: dataPath, hasHeader: true, separatorChar: ',');

            //Создаем преобразование для обнаружения аномальных значений
            var iidChangePointTransform = anomalyPointEstimator.Fit(emptyDataView);

            var transformedData = iidChangePointTransform.Transform(productSales);

            //Преобразовываем transformedData в строго типизированный IEnumerable для более удобного отображения
            var predictions = context.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            foreach (var p in predictions)
            {
                var results = new StringBuilder();
                foreach (var predict in p.Prediction)
                    results.Append($"{predict:f2}\t");

                if (p.Prediction[0] == 1)
                {
                    results.Append($" <-- {message}");
                }

                Console.WriteLine(results.ToString());
            }
            Console.WriteLine("");
        }
    }
}
