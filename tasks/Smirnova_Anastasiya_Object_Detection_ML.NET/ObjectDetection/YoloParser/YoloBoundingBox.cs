using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;

namespace ObjectDetection.YoloParser
{
    //BoundingBoxDimensions, который наследует от класса DimensionsBase, 
    //чтобы вместить размеры соответствующего ограничивающего прямоугольника.

    public class BoundingBoxDimensions : DimensionsBase { }

    //класс для ограничивающих прямоугольников.
    class YoloBoundingBox
    {
        //содержит размеры ограничивающего прямоугольника.
        public BoundingBoxDimensions Dimensions { get; set; }

        //содержит класс объекта, обнаруженного в ограничивающем прямоугольнике.
        public string Label { get; set; }

        //содержит достоверность класса.
        public float Confidence { get; set; }

        //содержит прямоугольное представление измерений ограничивающего прямоугольника.
        public RectangleF Rect
        {
            get { return new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
        }

        //содержит цвет, связанный с соответствующим классом, 
        //который используется для рисования изображения.
        public Color BoxColor { get; set; }
    }
}
