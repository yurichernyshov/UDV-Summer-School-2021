using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace AutoMLApp.DataStructures
{
    class SentimentIssue
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
