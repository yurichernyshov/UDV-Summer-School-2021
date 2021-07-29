using Microsoft.ML.Data;

namespace AutoML_Regression.DataStructures
{
    class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
