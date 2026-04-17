using OpenCvSharp;

using RapidOCRSharpOnnx.Configurations;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public class RecPreprocess: IRecPreprocess
    {
        private RecognizerConfig _recConfig;
        public RecPreprocess(RecognizerConfig recConfig)
        {
            _recConfig = recConfig;
        }
        public int ResizeNormImg(Mat img, int idx, float[] inputData, float max_wh_ratio)
        {
            // 获取原图尺寸和通道数
            int h = img.Height;
            int w = img.Width;
            int channels = img.Channels();
            int img_c = _recConfig.RecImgShape[0];
            int img_h = _recConfig.RecImgShape[1];
            int img_w = _recConfig.RecImgShape[2];

            if (img_c != channels)
                throw new ArgumentException($"The count of image channels does not match：expect {img_c}，actual {channels}");

            int img_ww = (int)(img_h * max_wh_ratio);

            // 计算缩放后的宽度（保持宽高比，但不超过目标宽度）
            float ratio = (float)w / h;
            double estimatedWidth = Math.Ceiling(img_h * ratio);

            int resized_w = estimatedWidth > img_ww ? img_ww : (int)estimatedWidth;

            // 缩放图像到 (resized_w, img_h)
            using Mat resized = new Mat();
            Cv2.Resize(img, resized, new OpenCvSharp.Size(resized_w, img_h));

            for (int i = 0; i < img_c; i++)
            {
                for (int j = 0; j < img_h; j++)
                {
                    for (int k = 0; k < img_ww; k++)
                    {
                        if (k < resized_w)
                        {
                            var val = (float)resized.At<Vec3b>(j, k)[i];
                            val = (val / 255.0f) * 2f - 1f;
                            inputData[idx++] = val;
                        }
                        else
                        {
                            inputData[idx++] = 0.0f;
                        }
                    }
                }
            }
            return idx;
        }
    }
}
