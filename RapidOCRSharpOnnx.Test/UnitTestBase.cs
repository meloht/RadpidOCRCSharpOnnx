using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Test
{
    public class UnitTestBase
    {
        protected const string detectModelName = "ch_PP-OCRv5_det_mobile.onnx";
        protected const string clsModelMobileName = "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";
        protected const string clsModelServerName = "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx";
        protected const string recModelName = "ch_PP-OCRv5_rec_mobile.onnx";
        protected const string png_txt = "txt.png";
        protected const string png_en = "en_txt.png";
        protected const string png_testClspng = "test_cls.png";
        protected const string png_textVerticalWords = "text_vertical_words.png";

        protected const string Res_txt = "Let's start collaborating Select a chat or channel from the list to get started.";
        protected const string Res_en = "She walks in beauty, like the night Of  cloudless climes and starry skies; And all that's best of dark and bright Meet in her aspect and l her eyes; Thus mellowed to that tender light Which heaven to gaudy day denies.";
        protected const string Res_testCls = "我 ";
        protected const string Res_textVerticalWords = "已取之時不參一人 是非不得間之人要取之 有評是是非非之士師也 見 而";

        protected string detectPath;
        protected string clsMobilePath;
        protected string clsServerPath;
        protected string recPath;

        public UnitTestBase()
        {
            detectPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", detectModelName);
            clsMobilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", clsModelMobileName);
            clsServerPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", clsModelServerName);
            recPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", recModelName);
        }

        protected string GetFullPath(string img)
        {
            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages", img);
        }

    }
}
