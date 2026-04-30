using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.TestCommon;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.TestGPU
{
    public class UnitTestDetClsRec : UnitTestBase, IDisposable
    {
        RapidOCRSharp _ocr;
        private int _deviceId;
        public UnitTestDetClsRec() : base()
        {
            _deviceId = Utils.GetMainGPU();
            _ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsMobilePath), _deviceId));
        }

        public void Dispose()
        {
            _ocr.Dispose();
        }

        [Fact]
        public void Test01()
        {
            var res = _ocr.RecognizeText(GetFullPath(png_txt));
            Assert.NotNull(res.TextBlocks);
            Assert.Equal(Res_txt, res.TextBlocks);
        }

        [Fact]
        public void Test02()
        {
            var res = _ocr.RecognizeText(GetFullPath(png_en));
            Assert.NotNull(res.TextBlocks);
            Assert.Equal(Res_en, res.TextBlocks);
        }

        [Fact]
        public void Test03()
        {
            var res = _ocr.RecognizeText(GetFullPath(png_testClspng));
            Assert.NotNull(res.TextBlocks);
            Assert.Equal(Res_testCls, res.TextBlocks);
        }

        [Fact]
        public void Test04()
        {
            var res = _ocr.RecognizeText(GetFullPath(png_textVerticalWords));
            Assert.NotNull(res.TextBlocks);
            Assert.Equal(Res_textVerticalWords, res.TextBlocks);
        }
    }
}
