using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Providers
{
    public class ExecutionProviderCPU : ExecutionProvider
    {
        public ExecutionProviderCPU(OcrConfig ocrConfig) : base(ocrConfig)
        {
        }

        protected override SessionOptions BuildSessionOptions()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.EnableCpuMemArena = true;
            return sessionOptions;

        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.CPU;
        }
    }
}
