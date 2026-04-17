using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace RapidOCRSharpOnnx.Utils
{
    public static class UtilsHelper
    {

        public static bool IsChineseChar(char ch)
        {
            // 对应Python的Unicode范围判断：
            // \u4e00-\u9fff：汉字
            // \u3000-\u303f：CJK标点（。、“”《》等）
            // \uff00-\uffef：全角符号（，．！？【】等）
            return (ch >= '\u4e00' && ch <= '\u9fff')
                   || (ch >= '\u3000' && ch <= '\u303f')
                   || (ch >= '\uff00' && ch <= '\uffef');
        }

        public static bool HasChineseChar(string text)
        {
            // 防护空值，避免NullReferenceException
            if (string.IsNullOrEmpty(text))
            {
                return false;
            }

            // LINQ的Any()等价于Python的any()：遍历每个字符，只要有一个满足就返回true
            return text.Any(ch => IsChineseChar(ch));
        }








   
    }
}
