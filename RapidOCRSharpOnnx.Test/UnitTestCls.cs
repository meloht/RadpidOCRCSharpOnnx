using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;

namespace RapidOCRSharpOnnx.Test
{
    public class UnitTestCls
    {
        const string detectModelName = "ch_PP-OCRv5_det_mobile.onnx";
        const string clsModelMobileName = "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";
        const string clsModelServerName = "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx";
        const string recModelName = "ch_PP-OCRv5_rec_mobile.onnx";
        private string detectPath;
        private string clsMobilePath;
        private string clsServerPath;
        private string recPath;
        public UnitTestCls()
        {
            detectPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", detectModelName);
            clsMobilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", clsModelMobileName);
            clsServerPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", clsModelServerName);
            recPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", recModelName);
        }

        [Fact]
        public void TestMobile()
        {
            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderCPU(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsMobilePath)));
            var res = ocr.RecognizeText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages", "test_cls.png"));
            Assert.NotNull(res.ClsResult.Data);
            Assert.True(res.ClsResult.Data.Length > 0);
            Assert.Equal("180", res.ClsResult.Data[0].Label);
        }

        [Fact]
        public void TestServer()
        {
            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderCPU(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsServerPath)));
            var res = ocr.RecognizeText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages", "test_cls.png"));
            Assert.NotNull(res.ClsResult.Data);
            Assert.True(res.ClsResult.Data.Length > 0);
            Assert.Equal("180", res.ClsResult.Data[0].Label);
        }
    }
}
