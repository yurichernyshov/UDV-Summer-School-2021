using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ProductSalesAnomalyDetection.Models
{
    class ProductSalesPrediction
    {
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
