using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Providers
{
    public class ExecutionProviderCoreML : ExecutionProvider
    {
        private CoreMLFlags _coreMLFlags;
        public ExecutionProviderCoreML(OcrConfig ocrConfig, CoreMLFlags coreMLFlags = CoreMLFlags.COREML_FLAG_USE_NONE) : base(ocrConfig)
        {
            _coreMLFlags = coreMLFlags;
        }

        protected override SessionOptions BuildSessionOptions()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.EnableCpuMemArena = true;
            sessionOptions.AppendExecutionProvider_CoreML(_coreMLFlags);
            return sessionOptions;
        }
      

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.CPU;
        }
    }
}
