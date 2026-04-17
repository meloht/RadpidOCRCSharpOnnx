using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference
{
    public interface IOcrDetector : IDisposable
    {
        DetectResult TextDetect(Mat image);
    }
}
