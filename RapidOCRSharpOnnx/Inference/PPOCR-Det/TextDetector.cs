using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class TextDetector : IDisposable
    {
        private InferenceSession _inferenceSession;
        private const int _minSize = 3;
        private const int _BOX_SORT_Y_THRESHOLD = 10;
        private DetPreprocess _detPreprocess;

        public TextDetector(string modelPath)
        {
            _inferenceSession = new InferenceSession(modelPath);
            _detPreprocess = new DetPreprocess();
        }
        public DetectResult Run(Mat image)
        {
            return null;
        }

        public void Dispose()
        {
            _inferenceSession?.Dispose();
        }
    }
}
