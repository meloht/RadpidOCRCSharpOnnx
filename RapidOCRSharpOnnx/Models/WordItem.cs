using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Models
{
    public class WordItem
    {
        public char[] Words { get; set; }

        public int[] WordCols { get; set; }

        public WordType WordType { get; set; }

        public float[] Confs { get; set; }
    }
}
