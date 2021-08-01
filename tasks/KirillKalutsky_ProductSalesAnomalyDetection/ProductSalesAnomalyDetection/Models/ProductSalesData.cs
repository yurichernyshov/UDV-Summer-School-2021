using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ProductSalesAnomalyDetection.Models
{
    class ProductSalesData
    {
        [LoadColumn(0)]
        public string Date { get; set; }

        [LoadColumn(1)]
        public float ProductSales { get; set; }
    }
}
