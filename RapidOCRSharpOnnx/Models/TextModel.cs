using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Models
{
    public class TextModel
    {
        public Point2f[] Boxes { get; set; }

        public string Text { get; set; }

        public float Confidence { get; set; }

        public TextModel(Point2f[] boxes, string text, float confidence)
        {
            Boxes = boxes;
            Text = text;
            Confidence = confidence;
        }

        public override string ToString()
        {
            return $"Text: {Text}, Confidence: {Confidence}, Boxes: [{string.Join(", ", Boxes.Select(p => $"({p.X}, {p.Y})"))}]";
        }

    }
}
