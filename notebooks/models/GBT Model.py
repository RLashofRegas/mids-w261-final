# Databricks notebook source
# MAGIC %md
# MAGIC # GBT Model

# COMMAND ----------

# package imports
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
from pyspark.sql import types
from pyspark.sql.functions import col, lag, udf, to_timestamp, monotonically_increasing_id
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, timedelta
from pyspark.ml.feature import IndexToString, StringIndexer, OneHotEncoder, VectorAssembler, Bucketizer, StandardScaler
import pandas as pd

# COMMAND ----------

# initialize the sql context
sqlContext = SQLContext(sc)

# COMMAND ----------

# global variables

# shared directory for our team (make sure it exists)
final_project_path = "dbfs:/mnt/mids-w261/group_5/"
dbutils.fs.mkdirs(final_project_path)

# output paths
train_data_output_path = final_project_path + "training_data_output/train.parquet"
test_data_output_path = final_project_path + "training_data_output/test.parquet"
train_data_output_path_one_hot = final_project_path + "training_data_output/train_one_hot.parquet"
test_data_output_path_one_hot = final_project_path + "training_data_output/test_one_hot.parquet"

# COMMAND ----------

# MAGIC %md ### GBT Model

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# Read in parquet file
train_GBT = spark.read.parquet(train_data_output_path)
test_GBT = spark.read.parquet(test_data_output_path)

# COMMAND ----------

train_GBT.printSchema()

# COMMAND ----------

# Example
display(train_GBT.sample(False, 0.0001))

# COMMAND ----------

# Assemble categorical and numeric features into vector
categorical = ["month", "day_of_week", "op_unique_carrier", "Holiday", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS", "origin_WND_type_code", "origin_CIG_ceiling_visibility_okay", "origin_VIS_variability", "dest_WND_type_code", "dest_CIG_ceiling_visibility_okay", "dest_VIS_variability", "crs_dep_hour"]

categorical_index = [i + "_Index" for i in categorical]

numeric = ["origin_num_flights","origin_avg_dep_delay", "origin_pct_dep_del15", "origin_avg_taxi_time", "origin_avg_weather_delay", "origin_avg_nas_delay", "origin_avg_security_delay", "origin_avg_late_aircraft_delay", "dest_num_flights","dest_avg_dep_delay", "dest_pct_dep_del15", "dest_avg_taxi_time", "dest_avg_weather_delay", "dest_avg_nas_delay", "dest_avg_security_delay", "dest_avg_late_aircraft_delay", "carrier_num_flights", "carrier_avg_dep_delay", "carrier_avg_carrier_delay", "origin_WND_direction_angle", "origin_WND_speed_rate", "origin_CIG_ceiling_height", "origin_VIS_distance", "origin_TMP_air_temperature", "origin_DEW_dew_point_temp", "origin_SLP_sea_level_pressure", "dest_WND_direction_angle", "dest_WND_speed_rate", "dest_CIG_ceiling_height", "dest_VIS_distance", "dest_TMP_air_temperature", "dest_DEW_dew_point_temp", "dest_SLP_sea_level_pressure"]

features = categorical_index + numeric
assembler = VectorAssembler(inputCols=features, outputCol="features").setHandleInvalid("skip")

train_GBT = assembler.transform(train_GBT)
test_GBT = assembler.transform(test_GBT)

# COMMAND ----------

# Define GBT model
gbt = GBTClassifier(labelCol="label", featuresCol="features")

# COMMAND ----------

# Chain indexers and GBT in a pipeline
# Can use if we index all data before split
# pipeline = Pipeline(stages=[labelIndexer, StringIndexer, gbt])

# COMMAND ----------

# Define a grid of hyperparameters to test:
#  - maxDepth: max depth of each decision tree in the GBT ensemble
#  - maxIter: iterations, i.e., number of trees in each GBT ensemble
# In this example notebook, we keep these values small.  In practice, to get the highest accuracy, you would likely want to try deeper trees (10 or higher) and more trees in the ensemble (>100).

paramGrid = ParamGridBuilder().build()
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()

# COMMAND ----------

# options for classification evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label")

# COMMAND ----------

# Cross validation?
# number of folds?
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds = 10)

# COMMAND ----------

# Train GBT model with cross validation
GBT_model = cv.fit(train_GBT)

# COMMAND ----------

# Make predictions
predictions = GBT_model.transform(validation_GBT)

# COMMAND ----------

# display sample of labels and predictions
labelAndPrediction = predictions.select("label", "prediction", "features")
display(labelAndPrediction.sample(False, 0.0001))

# COMMAND ----------

# accuracy
misclassified = labelAndPrediction.where(labelAndPrediction.label != labelAndPrediction.prediction).count()
misclassified_percent = misclassified/labelAndPrediction.count()
print('Accuracy: {:.3f}'.format(1-misclassified_percent))

# precision, recall
TP = labelAndPrediction.where((labelAndPrediction.label == 1) & (labelAndPrediction.prediction == 1)).count()
FP = labelAndPrediction.where((labelAndPrediction.label == 0) & (labelAndPrediction.prediction == 1)).count()
TN = labelAndPrediction.where((labelAndPrediction.label == 0) & (labelAndPrediction.prediction == 0)).count()
FN = labelAndPrediction.where((labelAndPrediction.label == 1) & (labelAndPrediction.prediction == 0)).count()

print('Recall: {:.3f}'.format(TP/(TP + FN)))
print('Precision: {:.3f}'.format(TP/(TP + FP)))

# COMMAND ----------

# what metric is this??
evaluator.evaluate(predictions)

# COMMAND ----------

auc_roc = test_GBT.avgMetrics
print(auc_roc)

# COMMAND ----------

test_GBT.bestModel.featureImportances

# COMMAND ----------

# or use indexing of feature vector
for i in range(len(features)):
    print("{}: {}".format(features[i],round(TEST_GBT_model.bestModel.featureImportances[i],3)))