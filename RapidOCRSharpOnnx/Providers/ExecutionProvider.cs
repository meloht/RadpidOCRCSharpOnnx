using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Providers
{
    public abstract class ExecutionProvider
    {
        protected abstract IOcrDetector GetDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess);
        protected abstract IOcrClassifier GetClassifier(InferenceSession session, SessionOptions options);

        protected abstract IOcrRecognizer GetRecognizer(InferenceSession session, SessionOptions options);
        protected abstract DeviceType GetDeviceType();

        public OcrConfig OcrConfig { get; private set; }

        public ExecutionProvider(OcrConfig ocrConfig)
        {
            OcrConfig = ocrConfig;
        }

       
    }
}
