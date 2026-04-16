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
    public class ExecutionProviderCPU : ExecutionProvider, IExecutionProvider
    {
        public ExecutionProviderCPU(OcrConfig ocrConfig) : base(ocrConfig)
        {
        }

        public IOcrDetector CreateDetector()
        {
            throw new NotImplementedException();
        }

        protected override IOcrClassifier GetClassifier(InferenceSession session, SessionOptions options)
        {
            throw new NotImplementedException();
        }

        protected override IOcrDetector GetDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess)
        {
            throw new NotImplementedException();
        }

        protected override DeviceType GetDeviceType()
        {
            throw new NotImplementedException();
        }

        protected override IOcrRecognizer GetRecognizer(InferenceSession session, SessionOptions options)
        {
            throw new NotImplementedException();
        }
    }
}
