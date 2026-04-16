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
    public class ExecutionProviderCoreML : ExecutionProvider, IExecutionProvider
    {
        private CoreMLFlags _coreMLFlags;
        public ExecutionProviderCoreML(OcrConfig ocrConfig, CoreMLFlags coreMLFlags = CoreMLFlags.COREML_FLAG_USE_NONE) : base(ocrConfig)
        {
            _coreMLFlags = coreMLFlags;
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
