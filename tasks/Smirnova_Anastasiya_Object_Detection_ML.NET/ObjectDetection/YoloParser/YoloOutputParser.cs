using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace ObjectDetection.YoloParser
{
    class CellDimensions : DimensionsBase { }

    //средство анализа.
    class YoloOutputParser
    {
        public const int ROW_COUNT = 13;
        public const int COL_COUNT = 13;
        //задает общее число значений, содержащихся в одной ячейке сетки.
        public const int CHANNEL_COUNT = 125;
        public const int BOXES_PER_CELL = 5;
        //число компонентов, содержащихся в поле (X, Y, высота, ширина, достоверность).
        public const int BOX_INFO_FEATURE_COUNT = 5;
        //число прогнозов класса, содержащихся в каждом ограничивающем прямоугольнике.
        public const int CLASS_COUNT = 20;
        public const float CELL_WIDTH = 32;
        public const float CELL_HEIGHT = 32;

        private int channelStride = ROW_COUNT * COL_COUNT;

        //список привязок для всех пяти ограничивающих прямоугольников.
        //Привязки — это предварительно определенные коэффициенты высоты и ширины
        //ограничивающих прямоугольников.Поскольку набор данных известен и значения предварительно вычислены,
        //привязки могут быть жестко запрограммированы.
        private float[] anchors = new float[]
        { 
            1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
        };

        //модель прогнозирует 20 классов, которые являются подмножеством 
        //общего числа классов, прогнозируемых исходной моделью YOLOv2.
        private string[] labels = new string[]
        {
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };

        //С каждым из классов связаны цвета.
        private static Color[] classColors = new Color[]
        {
            Color.Khaki,
            Color.Fuchsia,
            Color.Silver,
            Color.RoyalBlue,
            Color.Green,
            Color.DarkOrange,
            Color.Purple,
            Color.Gold,
            Color.Red,
            Color.Aquamarine,
            Color.Lime,
            Color.AliceBlue,
            Color.Sienna,
            Color.Orchid,
            Color.Tan,
            Color.LightPink,
            Color.Yellow,
            Color.HotPink,
            Color.OliveDrab,
            Color.SandyBrown,
            Color.DarkTurquoise
        };

        //Sigmoid применяет функцию-сигмоиду, которая выводит число от 0 до 1.
        private float Sigmoid(float value)
        {
            var k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        //Softmax нормализует входной вектор в распределение вероятности.
        private float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        //GetOffset сопоставляет элементы в выходных данных одномерной модели
        //с соответствующей позицией в тензоре 125 x 13 x 13.
        private int GetOffset(int x, int y, int channel)
        {
            // YOLO outputs a tensor that has a shape of 125x13x13, which 
            // WinML flattens into a 1D array.  To access a specific channel 
            // for a given (x,y) cell position, we need to calculate an offset
            // into the array
            return (channel * this.channelStride) + (y * COL_COUNT) + x;
        }

        //ExtractBoundingBoxes извлекает измерения ограничивающего 
        //прямоугольника с помощью метода GetOffset из выходных данных модели.
        private BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
        {
            return new BoundingBoxDimensions
            {
                X = modelOutput[GetOffset(x, y, channel)],
                Y = modelOutput[GetOffset(x, y, channel + 1)],
                Width = modelOutput[GetOffset(x, y, channel + 2)],
                Height = modelOutput[GetOffset(x, y, channel + 3)]
            };
        }

        //GetConfidence извлекает значение достоверности того,
        //что модель обнаружила объект, и использует функцию Sigmoid, чтобы преобразовать ее в процент.
        private float GetConfidence(float[] modelOutput, int x, int y, int channel)
        {
            return Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]);
        }

        //MapBoundingBoxToCell использует измерения ограничивающего прямоугольника
        //и сопоставляет их с соответствующей ячейкой на изображении.
        private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
        {
            return new CellDimensions
            {
                X = ((float)x + Sigmoid(boxDimensions.X)) * CELL_WIDTH,
                Y = ((float)y + Sigmoid(boxDimensions.Y)) * CELL_HEIGHT,
                Width = (float)Math.Exp(boxDimensions.Width) * CELL_WIDTH * anchors[box * 2],
                Height = (float)Math.Exp(boxDimensions.Height) * CELL_HEIGHT * anchors[box * 2 + 1],
            };
        }

        //ExtractClasses извлекает прогнозы класса для ограничивающего прямоугольника
        //из выходных данных модели с помощью метода GetOffset и превращает их в распределение
        //вероятности с помощью метода Softmax.
        public float[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
        {
            float[] predictedClasses = new float[CLASS_COUNT];
            int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;
            for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
            }
            return Softmax(predictedClasses);
        }

        //GetTopResult выбирает из списка прогнозируемых классов класс с наибольшей вероятностью.
        private ValueTuple<int, float> GetTopResult(float[] predictedClasses)
        {
            return predictedClasses
                .Select((predictedClass, index) => (Index: index, Value: predictedClass))
                .OrderByDescending(result => result.Value)
                .First();
        }

        //IntersectionOverUnion фильтрует перекрывающиеся ограничивающие прямоугольники с более низкими вероятностями.
        private float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
        {
            var areaA = boundingBoxA.Width * boundingBoxA.Height;

            if (areaA <= 0)
                return 0;

            var areaB = boundingBoxB.Width * boundingBoxB.Height;

            if (areaB <= 0)
                return 0;

            var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
            var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
            var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
            var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }

        //ParseOutputs для обработки выходных данных, создаваемых моделью
        public IList<YoloBoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .3F)
        {
            //список для хранения ограничивающих прямоугольников
            var boxes = new List<YoloBoundingBox>();

            //Каждое изображение делится на сетку из ячеек 13 x 13. 
            //Каждая ячейка содержит пять ограничивающих прямоугольников.
            //Код для обработки всех полей в каждой ячейке.
            for (int row = 0; row < ROW_COUNT; row++)
            {
                for (int column = 0; column < COL_COUNT; column++)
                {
                    for (int box = 0; box < BOXES_PER_CELL; box++)
                    {
                        //вычислить начальную точку текущего поля в выходных данных одномерной модели.
                        var channel = (box * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT));

                        //получить размеры текущего ограничивающего прямоугольника.
                        BoundingBoxDimensions boundingBoxDimensions = ExtractBoundingBoxDimensions(yoloModelOutputs, row, column, channel);
                        
                        //получить достоверность для текущего ограничивающего прямоугольника.
                        float confidence = GetConfidence(yoloModelOutputs, row, column, channel);
                        
                        //связать текущий ограничивающий прямоугольник с текущей обрабатываемой ячейкой.
                        CellDimensions mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxDimensions);

                        if (confidence < threshold)
                            continue;

                        //получение вероятности распределения прогнозируемых классов 
                        //для текущего ограничивающего прямоугольника с помощью метода ExtractClasses
                        float[] predictedClasses = ExtractClasses(yoloModelOutputs, row, column, channel);

                        //получить значение и индекс класса с наибольшей вероятностью
                        //для текущего поля и вычислить его оценку.
                        var (topResultIndex, topResultScore) = GetTopResult(predictedClasses);
                        var topScore = topResultScore * confidence;

                        //topScore, чтобы снова оставить только ограничивающие
                        //прямоугольники выше указанного порогового значения.
                        if (topScore < threshold)
                            continue;

                        //если текущий ограничивающий прямоугольник превышает пороговое значение,
                        //создаем новый объект BoundingBox и добавляем его в список boxes
                        boxes.Add(new YoloBoundingBox()
                        {
                            Dimensions = new BoundingBoxDimensions
                            {
                                X = (mappedBoundingBox.X - mappedBoundingBox.Width / 2),
                                Y = (mappedBoundingBox.Y - mappedBoundingBox.Height / 2),
                                Width = mappedBoundingBox.Width,
                                Height = mappedBoundingBox.Height,
                            },
                            Confidence = topScore,
                            Label = labels[topResultIndex],
                            BoxColor = classColors[topResultIndex]
                        });
                    }
                }
            }

            return boxes;
        }

        //для удаления перекрывающихся изображений необходимо провести дополнительную фильтрацию.
        public IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
        {
            //создания массива, который равен размеру обнаруженных полей,
            //помечая все слоты как активные или готовые к обработке.
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];

            for (int i = 0; i < isActiveBoxes.Length; i++)
                isActiveBoxes[i] = true;

            //отсортируем список, содержащий ограничивающие прямоугольники, в порядке убывания.
            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                    .OrderByDescending(b => b.Box.Confidence)
                    .ToList();

            //список для хранения отфильтрованных результатов.
            var results = new List<YoloBoundingBox>();

            for (int i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    //Если результаты больше указанного предельного числа полей для извлечения,
                    //следует выйти из цикла
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    if (results.Count >= limit)
                        break;
                    //В противном случае просмотрим соседние ограничивающие прямоугольники
                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        //если смежное поле активно или готово к обработке,
                        //используем метод IntersectionOverUnion, чтобы проверить,
                        //превышают ли первое и второе поля указанное пороговое значение.
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;

                            if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }

                    if (activeCount <= 0)
                        break;
                }
            }

            return results;

        }
    }
}
