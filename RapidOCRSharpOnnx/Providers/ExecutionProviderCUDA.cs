using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Providers
{
    public class ExecutionProviderCUDA : ExecutionProvider
    {
        private int _deviceId;
        private Dictionary<string, string> _providerOptionsDict;

        public ExecutionProviderCUDA(OcrConfig ocrConfig, int deviceId = 0, Dictionary<string, string> providerOptionsDict = null) : base(ocrConfig)
        {
            _deviceId = deviceId;
            _providerOptionsDict = providerOptionsDict;
        }

        protected override SessionOptions BuildSessionOptions()
        {
            SessionOptions options;
            if (this._providerOptionsDict != null && this._providerOptionsDict.Count > 0)
            {
                if (_providerOptionsDict.ContainsKey("device_id"))
                {
                    _providerOptionsDict["device_id"] = _deviceId.ToString();
                }
                else
                {
                    _providerOptionsDict.Add("device_id", _deviceId.ToString());
                }
                var cudaProviderOptions = new OrtCUDAProviderOptions();
                cudaProviderOptions.UpdateOptions(_providerOptionsDict);
                options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
            }
            else
            {
                options = SessionOptions.MakeSessionOptionWithCudaProvider(_deviceId);
            }

            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.EnableCpuMemArena = true;
            return options;
        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.GPU;
        }
    }
}
