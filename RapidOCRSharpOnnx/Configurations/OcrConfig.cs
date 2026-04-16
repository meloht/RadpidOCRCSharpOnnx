using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Configurations
{
    public class OcrConfig
    {
        public float TextScore { get; set; } = 0.5f;
        public int MinHeight { get; set; } = 30;
        public int WidthHeightRatio { get; set; } = 8;
        public int MaxSideLen { get; set; } = 2000;
        public int MinSideLen { get; set; } = 30;
        public bool ReturnWordBox { get; set; } = false;
        public bool ReturnSingleCharBox { get; set; } = false;

        public DetectorConfig DetectorConfig { get; set; }

        public ClassifierConfig ClassifierConfig { get; set; }

        public RecognizerConfig RecognizerConfig { get; set; }
    }
}
