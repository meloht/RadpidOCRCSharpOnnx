using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public interface IClsPreprocess
    {
        int ResizeNormImg(Mat img, int idx, float[] inputData, int[] clsImageShape);
    }
}
