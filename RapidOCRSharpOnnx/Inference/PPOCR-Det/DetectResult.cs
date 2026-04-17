using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class DetectResult
    {
        public Point2f[][] Boxes { get; set; }
        public Mat[] ImgCropList { get; set; }
        public float RatioH { get; set; }
        public float RatioW { get; set; }

        public int PaddingTop { get; set; }
        public int PaddingLeft { get; set; }

        public DetectResult(Point2f[][] boxes, Mat[] imgCropList)
        {
            Boxes = boxes;
            ImgCropList = imgCropList;
        }
    }
}
