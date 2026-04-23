using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models
{
    public record ClsPreResultBatch(OcrBatchResult BatchResult, float[] InputData, Mat img);

}
