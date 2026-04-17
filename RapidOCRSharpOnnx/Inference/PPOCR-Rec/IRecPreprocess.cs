using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public interface IRecPreprocess
    {
        int ResizeNormImg(Mat img, int idx, float[] inputData, float max_wh_ratio);
    }
}
