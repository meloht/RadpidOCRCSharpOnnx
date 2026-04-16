using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference;
using RapidOCRSharpOnnx.Providers;
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

        public OcrConfig OcrConfig { get => _executionProvider.OcrConfig; }


        public RapidOCRSharp(IExecutionProvider executionProvider)
        {
            _executionProvider = executionProvider;
        }

        public void Dispose()
        {
            _ocrDetector?.Dispose();
            _ocrClassifier?.Dispose();
            _ocrRecognizer?.Dispose();
        }
    }
}
