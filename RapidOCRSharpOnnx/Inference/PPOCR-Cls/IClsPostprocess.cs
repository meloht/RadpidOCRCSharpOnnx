using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public interface IClsPostprocess
    {
        void ClsPostProcess(OrtValue ortValue, int ij, DisposableList<Mat> imgList, ClsResult[] cls_res);
    }
}
