using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference
{
    public class ExecutePipeline : IExecutePipeline
    {
        private IOcrDetector _ocrDetector;
        private IOcrClassifier _ocrClassifier;
        private IOcrRecognizer _ocrRecognizer;
        protected OcrConfig _ocrConfig;
        public async Task<OcrBatchResult[]> BatchAsync(List<string> imageList)
        {
            Channel<OcrBatchResult> channelRecPre = Channel.CreateBounded<OcrBatchResult>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));

            OcrBatchResult[] batchResults = new OcrBatchResult[imageList.Count];
            List<Task> tasks = ExecuteTask(imageList, channelRecPre, batchResults);

            var consumerRec = Task.Run(async () =>
            {
                await foreach (OcrBatchResult item in channelRecPre.Reader.ReadAllAsync())
                {
                    await _ocrRecognizer.BatchRecAsync(item);
                }
            });

            tasks.Add(consumerRec);

            await Task.WhenAll(tasks);

            return batchResults;
        }

        private List<Task> ExecuteTask(List<string> imageList, Channel<OcrBatchResult> channelRecPre, OcrBatchResult[] batchResults)
        {
            Channel<OcrBatchResult> channelDetNext = channelRecPre;
            if (_ocrClassifier != null)
            {
                channelDetNext = Channel.CreateBounded<OcrBatchResult>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));
            }

            List<Task> tasks = new List<Task>();
            tasks.Add(_ocrDetector.BatchDetectAsync(imageList, channelDetNext.Writer, batchResults));

            if (_ocrClassifier != null)
            {
                var consumerCls = Task.Run(async () =>
                {
                    await foreach (OcrBatchResult item in channelDetNext.Reader.ReadAllAsync())
                    {

                        await _ocrClassifier.BatchClsAsync(item, channelDetNext, channelRecPre.Writer);
                    }
                });

                tasks.Add(consumerCls);
            }
            return tasks;
        }

        public async IAsyncEnumerable<OcrBatchResult> BatchForeachAsync(List<string> imageList)
        {
            Channel<OcrBatchResult> channelRecPre = Channel.CreateBounded<OcrBatchResult>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));
           
            OcrBatchResult[] batchResults = new OcrBatchResult[imageList.Count];

            List<Task> tasks = ExecuteTask(imageList, channelRecPre, batchResults);
           
            await foreach (OcrBatchResult item in channelRecPre.Reader.ReadAllAsync())
            {
                await _ocrRecognizer.BatchRecAsync(item);
                yield return item;
            }

            await Task.WhenAll(tasks);
        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }
    }
}
