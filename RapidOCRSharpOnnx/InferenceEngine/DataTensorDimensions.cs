using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.InferenceEngine
{
    public struct DataTensorDimensions
    {
        public float[] Data { get; private set; }
        public long[] Dimensions { get; private set; }
        public float RatioH { get; set; }
        public float RatioW { get; set; }

        public int PaddingTop { get; set; }
        public int PaddingLeft { get; set; }

        public DataTensorDimensions(float[] data, long[] dimensions, float ratioH = 1.0f, float ratioW = 1.0f, int paddingTop = 0, int paddingLeft = 0)
        {
            Data = data;
            Dimensions = dimensions;
            RatioH = ratioH;
            RatioW = ratioW;
            PaddingTop = paddingTop;
            PaddingLeft = paddingLeft;
        }
    }
}
