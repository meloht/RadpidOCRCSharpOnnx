using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Providers
{
    public class ExecutionProviderDirectML: ExecutionProvider
    {
        private int _deviceId;

        public ExecutionProviderDirectML(OcrConfig ocrConfig, int deviceId=0) : base(ocrConfig)
        {
            _deviceId = deviceId;
        }

        protected override SessionOptions BuildSessionOptions()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.AppendExecutionProvider_DML(this._deviceId);
            sessionOptions.EnableCpuMemArena = true;
            return sessionOptions;
        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.GPU;
        }
    }
}
