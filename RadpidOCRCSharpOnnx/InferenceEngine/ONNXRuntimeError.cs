using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.InferenceEngine
{
    public class ONNXRuntimeError : Exception
    {
        public ONNXRuntimeError(string message) : base(message)
        {

        }

    }
}
