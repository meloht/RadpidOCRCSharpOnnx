using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Providers
{
    public interface IExecutionProvider
    {
        IOcrDetector CreateDetector();

        IOcrClassifier CreateClassifier();

        IOcrRecognizer CreateRecognizer();

        OcrConfig OcrConfig { get; }

    }
}
