using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Providers;
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
        private IExecutionProvider _executionProvider;
        private IOcrDetector _ocrDetector;
        private IOcrClassifier _ocrClassifier;
        private IOcrRecognizer _ocrRecognizer;
        private OcrDrawerSkia _ocrDrawerSkia;
        protected OcrConfig _ocrConfig;

        public ExecutePipeline(OcrConfig ocrConfig, IExecutionProvider executionProvider)
        {
            _ocrConfig = ocrConfig;
            _executionProvider = executionProvider;
            _ocrDetector = _executionProvider.CreateDetector();
            _ocrClassifier = _executionProvider.CreateClassifier();
            _ocrRecognizer = _executionProvider.CreateRecognizer();
            _ocrDrawerSkia = new OcrDrawerSkia(_ocrConfig);
        }

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

        public OcrResult RecognizeText(string imagePath, string savePath = null)
        {
            ValidationUtils.ValidateImage(imagePath);
            using Mat image = Cv2.ImRead(imagePath);
            return RecognizeText(image, savePath);
        }
        public OcrResult RecognizeText(Mat image, string savePath = null)
        {
            OcrResult result = new OcrResult();
            var detResult = _ocrDetector.TextDetect(image);
            result.DetResult = detResult;
            using (detResult.Data.ImgCropList)
            {
                if (_ocrClassifier != null)
                {
                    var ClsResult = _ocrClassifier.TextClassify(detResult.Data.ImgCropList);
                    result.ClsResult = ClsResult;
                }

                var recResults = _ocrRecognizer.TextRecognize(detResult.Data.ImgCropList);
                result.RecResult = recResults;

                for (int i = 0; i < detResult.Data.DetItems.Length; i++)
                {
                    detResult.Data.DetItems[i].Word = recResults.Data[i].Label;
                }
                result.TextBlocks = string.Join(" ", recResults.Data.Select(r => r.Label));

                if (!string.IsNullOrEmpty(savePath))
                {
                    _ocrDrawerSkia.DrawTextBlock(image, savePath, detResult.Data, recResults.Data);
                }
            }

            return result;
        }

        public void Dispose()
        {
            _ocrDetector?.Dispose();
            _ocrClassifier?.Dispose();
            _ocrRecognizer?.Dispose();
            _ocrDrawerSkia?.Dispose();
        }
    }
}
