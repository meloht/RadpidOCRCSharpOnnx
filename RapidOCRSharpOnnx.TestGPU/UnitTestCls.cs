using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.TestCommon;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.TestGPU
{
    public class UnitTestCls : UnitTestBase
    {
        [Fact]
        public void TestMobile()
        {
            var deviceId = Utils.GetMainGPU();
            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsMobilePath), deviceId));
            var res = ocr.RecognizeText(GetFullPath(png_testClspng));
            Assert.NotNull(res.ClsResult.Data);
            Assert.True(res.ClsResult.Data.Length > 0);
            Assert.Equal("180", res.ClsResult.Data[0].Label);
        }

        [Fact]
        public void TestServer()
        {
            var deviceId = Utils.GetMainGPU();
            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsServerPath), deviceId));
            var res = ocr.RecognizeText(GetFullPath(png_testClspng));
            Assert.NotNull(res.ClsResult.Data);
            Assert.True(res.ClsResult.Data.Length > 0);
            Assert.Equal("180", res.ClsResult.Data[0].Label);
        }
    }
}
