using OpenCvSharp;
using OpenCvSharp.ML;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx
{
    public class RapidOCRSharp : IDisposable
    {
        private IExecutionProvider _executionProvider;
        private IOcrDetector _ocrDetector;
        private IOcrClassifier _ocrClassifier;
        private IOcrRecognizer _ocrRecognizer;
        private readonly TextCalRecBox _textCalRecBox;

        public OcrConfig OcrConfig
        {
            get { return _executionProvider.OcrConfig; }
        }


        public RapidOCRSharp(IExecutionProvider executionProvider)
        {
            _executionProvider = executionProvider;
            _textCalRecBox = new TextCalRecBox(OcrConfig);
            _ocrDetector = _executionProvider.CreateDetector();
            _ocrClassifier = _executionProvider.CreateClassifier();
            _ocrRecognizer = _executionProvider.CreateRecognizer();
        }

        public void DetAndRec(string imagePath)
        {
            using Mat image = Cv2.ImRead(imagePath);
            var detResult = _ocrDetector.TextDetect(image);
            var clsBoxes = _ocrClassifier.TextClassify(detResult.ImgCropList);
            var recResults = _ocrRecognizer.TextRecognize(detResult.ImgCropList);

            UtilsHelper.MapBoxesToOriginal(detResult.Boxes, detResult.RatioH, detResult.RatioW, detResult.PaddingTop, detResult.PaddingLeft, image.Height, image.Width);

            var croppedImgList = UtilsHelper.MapImgToOriginal(detResult.ImgCropList, detResult.RatioH, detResult.RatioW);
            var resCorp = _textCalRecBox.CalRecBoxes(croppedImgList, recResults, detResult.Boxes);

            //using var input = SKBitmap.Decode(imgPath);
            //return recognizedResults;
        }

        public void Dispose()
        {
            _ocrDetector?.Dispose();
            _ocrClassifier?.Dispose();
            _ocrRecognizer?.Dispose();
        }
    }
}
