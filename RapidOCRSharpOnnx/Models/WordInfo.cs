using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Models
{
    public class WordInfo
    {
        public float LineTxtLen { get; set; }
        public List<WordItem> WordItems { get; set; }
       
    }
}
