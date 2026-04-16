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
    public class ExecutionProviderOpenVINO : ExecutionProvider, IExecutionProvider
    {
        private const string CPU = "CPU";
        private const string GPU = "GPU";
        private const string GPU0 = "GPU.0";
        private const string GPU1 = "GPU.1";
        private const string NPU = "NPU";
        private IntelDeviceType _intelDeviceType;

        public ExecutionProviderOpenVINO(OcrConfig ocrConfig, IntelDeviceType intelDeviceType) : base(ocrConfig)
        {
            _intelDeviceType = intelDeviceType;
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

        private string GetIntelDeviceType()
        {
            switch (_intelDeviceType)
            {
                case IntelDeviceType.CPU:
                    return CPU;
                case IntelDeviceType.GPU:
                    return GPU;
                case IntelDeviceType.GPU0:
                    return GPU0;
                case IntelDeviceType.GPU1:
                    return GPU1;
                case IntelDeviceType.NPU:
                    return NPU;
                default:
                    return CPU;
            }
        }
    }
}
