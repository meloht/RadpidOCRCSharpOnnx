using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Providers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;
using static System.Runtime.InteropServices.JavaScript.JSType;


namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public abstract class TextDetectorBase : OnnxInferenceCore
    {

        protected IDetPreprocess _detPreprocess;
        protected IDetPostprocess _detPostprocess;


        public TextDetectorBase(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OcrConfig ocrConfig, DeviceType deviceType)
            : base(session, options, ocrConfig, deviceType)
        {
            _detPreprocess = preprocess;
            _detPostprocess = postprocess;
        }


        public ResultPerf<DetResult> TextDetect(Mat image)
        {
            PerfModel perf = new PerfModel();
            _stopwatch.Restart();
            using Mat resizedImg = image.Clone();
            var data = _detPreprocess.Preprocess(image, resizedImg);
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(data.Data, data.Dimensions);
            _stopwatch.Stop();
            perf.Preprocess += _stopwatch.ElapsedMilliseconds;

            using var output0 = InferenceRun(inputOrtValue, perf);
            using var ortValue = output0[0];

            _stopwatch.Restart();
            var res = _detPostprocess.PostProcess(resizedImg, ortValue);

            res.ResizeData = data.ResizeData;

            ResultPerf<DetResult> result = new ResultPerf<DetResult>();
            result.Data = res;
            result.Perf = perf;
            _stopwatch.Stop();
            perf.Postprocess += _stopwatch.ElapsedMilliseconds;
            perf.SumTotal();
            return result;
        }


        public async Task BatchDetectAsync(List<string> listImg, ChannelWriter<OcrBatchResult> nextChannelWriter, OcrBatchResult[] batchResults)
        {
            int idx = 0;
            Task[] tasks = new Task[listImg.Count + 2];
            Channel<DetPreResultBatch> channelDet = Channel.CreateBounded<DetPreResultBatch>(GetChannelOptions(_ocrConfig.BatchPoolSize));
            var producer = _detPreprocess.PreprocessBatchAsync(listImg, _deviceType, channelDet.Writer);

            tasks[idx] = producer;
            Interlocked.Increment(ref idx);

            var consumer = Task.Run(async () =>
            {
                await foreach (DetPreResultBatch item in channelDet.Reader.ReadAllAsync())
                {
                    using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(item.PreResult.Data, item.PreResult.Dimensions);

                    var output0 = InferenceRun(inputOrtValue, null);
                    tasks[idx] = BatchPostProcessAsync(output0, item, batchResults[idx - 1], nextChannelWriter);
                    Interlocked.Increment(ref idx);

                }
            });
            tasks[idx] = consumer;
            await Task.WhenAll(tasks);

            nextChannelWriter.Complete();
        }

        private async Task BatchPostProcessAsync(IDisposableReadOnlyCollection<OrtValue> output, DetPreResultBatch item, OcrBatchResult batchResult, ChannelWriter<OcrBatchResult> writer)
        {
            await Task.Run(async () =>
            {
                using (output)
                using (item.ResizedImg)
                {
                    using var ortValue = output[0];
                    var res = _detPostprocess.PostProcess(item.ResizedImg, ortValue);
                    res.ResizeData = item.PreResult.ResizeData;
                    batchResult.DetResult = res;
                    await writer.WriteAsync(batchResult);
                }
            });
        }

    }
}
