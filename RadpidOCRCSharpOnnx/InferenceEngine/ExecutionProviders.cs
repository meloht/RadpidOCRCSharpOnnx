using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace RadpidOCRCSharpOnnx.InferenceEngine
{
    public enum ExecutionProviders
    {
        /// <summary>
        /// CPUExecutionProvider
        /// </summary>
        [Description("CPUExecutionProvider")]
        CPU_EP,
        /// <summary>
        /// CUDAExecutionProvider
        /// </summary>
        [Description("CUDAExecutionProvider")]
        CUDA_EP,
        /// <summary>
        /// DmlExecutionProvider
        /// </summary>
        [Description("DmlExecutionProvider")]
        DirectML_EP,
      
        /// <summary>
        /// CoreMLExecutionProvider
        /// </summary>
        [Description("CoreMLExecutionProvider")]
        CoreML_EP
    }

    public static class EnumExtensions
    {
        public static string GetDescription(this Enum value)
        {
            var field = value.GetType().GetField(value.ToString());
            var attribute = (DescriptionAttribute)Attribute.GetCustomAttribute(field, typeof(DescriptionAttribute));
            return attribute == null ? value.ToString() : attribute.Description;
        }
    }
}
