using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Models
{
    public class OcrResult
    {
        public TextModel[] TextBlocks { get; set; }

        public PerfModel DetPerf { get; set; }

        public PerfModel ClsPerf { get; set; }

        public PerfModel RecPerf { get; set; }

        public override string ToString()
        {
            string res = "";
            if (TextBlocks != null && TextBlocks.Length > 0)
            {
                res = string.Join(" ", TextBlocks.Select(p => p.Text));

            }
            return $"TextBlocks: {res}, DetPerf: {DetPerf.TotalTime}ms, ClsPerf: {ClsPerf.TotalTime}ms, RecPerf: {RecPerf.TotalTime}ms";
        }

    }
}
