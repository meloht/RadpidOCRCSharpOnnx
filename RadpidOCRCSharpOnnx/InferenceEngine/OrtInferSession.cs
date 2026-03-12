using Microsoft.ML.OnnxRuntime;
using RadpidOCRCSharpOnnx.Config;
using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.InferenceEngine
{
    public class OrtInferSession : IDisposable
    {
        private InferenceSession _inferenceSession;
        private SessionOptions _sessionOptions;
        private ProviderConfig _providerConfig;

        public OrtInferSession(string modelPath)
        {
            _providerConfig = new ProviderConfig();
            _sessionOptions = InitSessionOptions();
            _inferenceSession = new InferenceSession(modelPath, _sessionOptions);
        }

        private SessionOptions InitSessionOptions()
        {
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            options.EnableCpuMemArena = OnnxEngineConfig.EnableCpuMemArena;
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            int cpu_nums = Environment.ProcessorCount;
            int intra_op_num_threads = OnnxEngineConfig.IntraOpNumThreads;
            if (intra_op_num_threads != -1 && 1 <= intra_op_num_threads && intra_op_num_threads <= cpu_nums)
            {
                options.IntraOpNumThreads = intra_op_num_threads;
            }

            int inter_op_num_threads = OnnxEngineConfig.InterOpNumThreads;
            if (inter_op_num_threads != -1 && 1 <= inter_op_num_threads && inter_op_num_threads <= cpu_nums)
            {
                options.InterOpNumThreads = inter_op_num_threads;
            }

            _providerConfig.SetProvider(options);

            return options;
        }



        public void Dispose()
        {
            _sessionOptions?.Dispose();
            _inferenceSession?.Dispose();

        }
    }
}
