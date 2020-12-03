# Databricks notebook source
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.functions import col, explode, array, lit

sqlContext = SQLContext(sc)

# COMMAND ----------

final_project_path = "dbfs:/mnt/mids-w261/group_5/"
dbutils.fs.mkdirs(final_project_path)

# COMMAND ----------

train_data_output_path = final_project_path + "training_data_output/train.parquet"
validation_data_output_path = final_project_path + "training_data_output/validation.parquet"
test_data_output_path = final_project_path + "training_data_output/test.parquet"
train_data_output_path_one_hot = final_project_path + "training_data_output/train_one_hot.parquet"
validation_data_output_path_one_hot = final_project_path + "training_data_output/validation_one_hot.parquet"
test_data_output_path_one_hot = final_project_path + "training_data_output/test_one_hot.parquet"

# COMMAND ----------

train_set = spark.read.option("header", "true").parquet(train_data_output_path_one_hot)
val_set = spark.read.option("header", "true").parquet(validation_data_output_path_one_hot)
test_set = spark.read.option("header", "true").parquet(test_data_output_path_one_hot)

# COMMAND ----------

# MAGIC %md ### Oversampling

# COMMAND ----------

major_df = train_set.filter(col("label") == 0)
minor_df = train_set.filter(col("label") == 1)
ratio = int(major_df.count()/minor_df.count())

a = range(ratio)
# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows 
train_set_oversampled = major_df.unionAll(oversampled_df)

# COMMAND ----------

# Index features

#Too many categories for airport IDs, aw1?
# categorical = ["month", "day_of_week", "op_unique_carrier", "origin_airport_id", "dest_airport_id", "Holiday", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS", "origin_CIG_ceiling_visibility_okay", "origin_VIS_variability", "origin_aw1_automated_atmospheric_condition", "dest_CIG_ceiling_visibility_okay", "dest_VIS_variability"]

#ALL
# categorical = ['origin_WND_direction_angle_Index', 'origin_VIS_variability_Index', 'PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index', 'dest_WND_direction_angle_Index', 'op_unique_carrier_Index',  "origin_airport_id", "dest_airport_id",'day_of_week_Index', 'dest_CIG_ceiling_visibility_okay_Index', 'dest_WND_type_code_Index', 'dest_VIS_variability_Index', 'Holiday_Index', 'origin_CIG_ceiling_visibility_okay_Index', 'crs_dep_hour_Index', 'month_Index', 'origin_WND_type_code_Index']

#REMOVE OP CARRIER INDEX and WIND DIRECTION ANGLE
categorical = ['PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index', 'origin_VIS_variability_Index', 'day_of_week_Index', 'dest_CIG_ceiling_visibility_okay_Index', 'dest_WND_type_code_Index', 'dest_VIS_variability_Index', 'Holiday_Index', 'origin_CIG_ceiling_visibility_okay_Index', 'crs_dep_hour_Index', 'month_Index', 'origin_WND_type_code_Index']

#NUMEREIC
#Full
numeric = ["origin_num_flights","origin_avg_dep_delay", "origin_pct_dep_del15", "origin_avg_taxi_time", "origin_avg_weather_delay", "origin_avg_nas_delay", "origin_avg_security_delay", "origin_avg_late_aircraft_delay", "dest_num_flights","dest_avg_dep_delay", "dest_pct_dep_del15", "dest_avg_taxi_time", "dest_avg_weather_delay", "dest_avg_nas_delay", "dest_avg_security_delay", "dest_avg_late_aircraft_delay", "carrier_num_flights", "carrier_avg_dep_delay", "carrier_avg_carrier_delay", "origin_WND_speed_rate", "origin_CIG_ceiling_height", "origin_VIS_distance", "origin_TMP_air_temperature", "origin_DEW_dew_point_temp", "dest_WND_speed_rate", "dest_CIG_ceiling_height", "dest_VIS_distance", "dest_TMP_air_temperature", "dest_DEW_dew_point_temp",
"origin_aa1_rain_depth", "dest_aa1_rain_depth", "origin_aj1_snow_depth", "dest_aj1_snow_depth"]

#Remove Engineered with many nulls
# numeric = ["origin_WND_speed_rate", "origin_CIG_ceiling_height", "origin_VIS_distance", "origin_TMP_air_temperature", "origin_DEW_dew_point_temp", "dest_WND_speed_rate", "dest_CIG_ceiling_height", "dest_VIS_distance", "dest_TMP_air_temperature", "dest_DEW_dew_point_temp", "origin_aa1_rain_depth", "dest_aa1_rain_depth", "origin_aj1_snow_depth", "dest_aj1_snow_depth"]




# COMMAND ----------

#INVESTIGATE % NULL
from pyspark.sql.functions import isnan, when, count, col
train_subset = train_set.select("origin_num_flights","origin_avg_dep_delay", "origin_pct_dep_del15", "origin_avg_taxi_time", "origin_avg_weather_delay", "origin_avg_nas_delay", "origin_avg_security_delay", "origin_avg_late_aircraft_delay", "dest_num_flights","dest_avg_dep_delay", "dest_pct_dep_del15", "dest_avg_taxi_time", "dest_avg_weather_delay", "dest_avg_nas_delay", "dest_avg_security_delay", "dest_avg_late_aircraft_delay", "carrier_num_flights", "carrier_avg_dep_delay", "carrier_avg_carrier_delay", "origin_WND_speed_rate", "origin_CIG_ceiling_height", "origin_VIS_distance", "origin_TMP_air_temperature", "origin_DEW_dew_point_temp", "dest_WND_speed_rate", "dest_CIG_ceiling_height", "dest_VIS_distance", "dest_TMP_air_temperature", "dest_DEW_dew_point_temp",
"origin_aa1_rain_depth", "dest_aa1_rain_depth", "origin_aj1_snow_depth", "dest_aj1_snow_depth")
display(train_subset.select([((count(when(col(c).isNull(), c)))/17247986).alias(c) for c in train_subset.columns]))

# COMMAND ----------

# # Assemble features into vector
features = categorical + numeric
assembler = VectorAssembler(inputCols=features, outputCol="features2")
assembler.setHandleInvalid("skip")

train_RF = assembler.transform(train_set)
train_RF_OS = assembler.transform(train_set_oversampled)
validation_RF = assembler.transform(val_set)
# test_RF = assembler.transform(test_RF)

# COMMAND ----------

#DROP DEP DEL15 = null
train_RF_drop_nulls = train_RF.where(f.col('dep_del15').isNotNull())
train_RF_OS_drop_nulls = train_RF_OS.where(f.col('dep_del15').isNotNull())
validation_RF_drop_nulls = validation_RF.where(f.col('dep_del15').isNotNull())

# COMMAND ----------

#Define rf model, Evaluator and paramgrid to use in CV
rf = RF(labelCol='label', featuresCol='features2', numTrees = 30, maxDepth = 10)

evaluator = BinaryClassificationEvaluator(labelCol="label")

paramGrid = ParamGridBuilder().build()
#       .addGrid(rf.numTrees, [int(x) for x in np.linspace(start =20, stop = 30, num = 2)]) \
#       .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 15, num = 2)]) \
#       .build()

# COMMAND ----------

#cross validation
cv = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)


# COMMAND ----------

# Train RF model
RF_model = cv.fit(train_RF_OS_drop_nulls)

# COMMAND ----------

#NOT USING CV
RF_model = rf.fit(train_RF_OS_drop_nulls)

# COMMAND ----------

# MAGIC %md Prediction and Evaluation

# COMMAND ----------

predictions = RF_model.transform(validation_RF_drop_nulls)
scoreAndLabels = predictions.select("prediction", "label")

# COMMAND ----------

scoreAndLabels.count()

# COMMAND ----------

scoreAndLabelsPandas = scoreAndLabels.toPandas()

import numpy as np
confusion_matrix = pd.crosstab(scoreAndLabelsPandas['label'], scoreAndLabelsPandas['prediction'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True,cmap='Blues', fmt='g')

plt.show()

# COMMAND ----------

#Area Under ROC
params = [{p.name: v for p, v in m.items()} for m in RF_model.getEstimatorParamMaps()]

pd.DataFrame.from_dict([
    {RF_model.getEvaluator().getMetricName(): metric, **ps} 
    for ps, metric in zip(params, RF_model.avgMetrics)
])


# COMMAND ----------

true_positive = scoreAndLabelsPandas[(scoreAndLabelsPandas.label == 1) & (scoreAndLabelsPandas.prediction == 1) ]['label'].count()
true_negative = scoreAndLabelsPandas[(scoreAndLabelsPandas.label == 0) & (scoreAndLabelsPandas.prediction == 0)]['label'].count()
false_positive = scoreAndLabelsPandas[(scoreAndLabelsPandas.label == 0) & (scoreAndLabelsPandas.prediction == 1)]['prediction'].count()
false_negative = scoreAndLabelsPandas[(scoreAndLabelsPandas.label == 1) & (scoreAndLabelsPandas.prediction == 0)]['label'].count()
accuracy = (true_positive + true_negative)/ (true_positive + true_negative + false_positive + false_negative)

precision = true_positive / (false_positive + true_positive)
recall = true_positive / (false_negative + true_positive)
f1 = (2 * precision * recall)/ (precision + recall)

print('''Precision: {}
Recall: {}
F1: {}
Accuracy: {}'''.format(precision, recall, f1, accuracy))

# COMMAND ----------

RF_model.featureImportances

# COMMAND ----------

# MAGIC %md #Small Data

# COMMAND ----------

# Read in parquet file
train_RF = spark.read.parquet(train_data_output_path)
validation_RF = spark.read.parquet(validation_data_output_path)

# COMMAND ----------

# Create test file without weather features
TEST_RF_train = train_RF.select("month", "day_of_week", "op_unique_carrier", "origin_airport_id", "dest_airport_id", "Holiday", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS","origin_num_flights","origin_avg_dep_delay", "origin_pct_dep_del15", "origin_avg_taxi_time", "origin_avg_weather_delay", "origin_avg_nas_delay", "origin_avg_security_delay", "origin_avg_late_aircraft_delay", "dest_num_flights","dest_avg_dep_delay", "dest_pct_dep_del15", "dest_avg_taxi_time", "dest_avg_weather_delay", "dest_avg_nas_delay", "dest_avg_security_delay", "dest_avg_late_aircraft_delay", "carrier_num_flights", "carrier_avg_dep_delay", "carrier_avg_carrier_delay", "dep_del15")

TEST_RF_val = validation_RF.select("month", "day_of_week", "op_unique_carrier", "origin_airport_id", "dest_airport_id", "Holiday", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS","origin_num_flights","origin_avg_dep_delay", "origin_pct_dep_del15", "origin_avg_taxi_time", "origin_avg_weather_delay", "origin_avg_nas_delay", "origin_avg_security_delay", "origin_avg_late_aircraft_delay", "dest_num_flights","dest_avg_dep_delay", "dest_pct_dep_del15", "dest_avg_taxi_time", "dest_avg_weather_delay", "dest_avg_nas_delay", "dest_avg_security_delay", "dest_avg_late_aircraft_delay", "carrier_num_flights", "carrier_avg_dep_delay", "carrier_avg_carrier_delay", "dep_del15")

# COMMAND ----------

# Index label
labelIndexer = StringIndexer(inputCol="dep_del15", outputCol="label")

# Set string indexer to handle nulls
labelIndexer.setHandleInvalid("skip")

TEST_RF_train = labelIndexer.fit(TEST_RF_train).transform(TEST_RF_train)
TEST_RF_val = labelIndexer.fit(TEST_RF_val).transform(TEST_RF_val)

# COMMAND ----------

# Index features
# Do we want to clean up join to remove more columns?
#features = [feature for feature in train_GBT.columns].remove("year", "quarter", "fl_date", "tail_num", "origin", "dep_del15")

categorical = ["month", "day_of_week", "op_unique_carrier", "Holiday", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS"]

categorical_index = [i + "_Index" for i in categorical]

# Index categorical variables
stringIndexer = StringIndexer(inputCols=categorical, outputCols=categorical_index)

# Set string indexer to handle nulls
stringIndexer.setHandleInvalid("skip")
    
numeric = ["origin_num_flights","origin_avg_dep_delay", "origin_pct_dep_del15", "origin_avg_taxi_time", "origin_avg_weather_delay", "origin_avg_nas_delay", "origin_avg_security_delay", "origin_avg_late_aircraft_delay", "dest_num_flights","dest_avg_dep_delay", "dest_pct_dep_del15", "dest_avg_taxi_time", "dest_avg_weather_delay", "dest_avg_nas_delay", "dest_avg_security_delay", "dest_avg_late_aircraft_delay", "carrier_num_flights", "carrier_avg_dep_delay", "carrier_avg_carrier_delay"]


TEST_RF_train = stringIndexer.fit(TEST_RF_train).transform(TEST_RF_train)
TEST_RF_val = stringIndexer.fit(TEST_RF_val).transform(TEST_RF_val)

# COMMAND ----------

# Assemble features into vector
features = categorical_index + numeric
assembler = VectorAssembler(inputCols=features, outputCol="features")

# Set assembler to handle nulls
assembler.setHandleInvalid("skip")

TEST_RF_train = assembler.transform(TEST_RF_train)
TEST_RF_val = assembler.transform(TEST_RF_val)

# COMMAND ----------

# Define RF model
rf = RF(labelCol='label', featuresCol='features',numTrees=20)

# COMMAND ----------

paramGrid = ParamGridBuilder().build()

# COMMAND ----------

# options for classification evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label")

# COMMAND ----------

cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds = 10)

# COMMAND ----------

# Train GBT model
TEST_RF_model = cv.fit(TEST_RF_train)

# COMMAND ----------

# MAGIC %md Prediction and Evaluation

# COMMAND ----------

#Area Under ROC
params = [{p.name: v for p, v in m.items()} for m in TEST_RF_model.getEstimatorParamMaps()]

pd.DataFrame.from_dict([
    {TEST_RF_model.getEvaluator().getMetricName(): metric, **ps} 
    for ps, metric in zip(params, TEST_RF_model.avgMetrics)
])


# COMMAND ----------

predictions = TEST_RF_model.transform(TEST_RF_val)


# COMMAND ----------

scoreAndLabels = predictions.select("prediction", "label")

# COMMAND ----------

# Let's use the run-of-the-mill evaluator
evaluator = BinaryClassificationEvaluator(labelCol='label')

# We have only two choices: area under ROC and PR curves :-(
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
auprc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
print("Area under ROC Curve: {:.4f}".format(auroc))
print("Area under PR Curve: {:.4f}".format(auprc))

# COMMAND ----------

scoreAndLabelsPandas = scoreAndLabels.toPandas()

# COMMAND ----------

import numpy as np
confusion_matrix = pd.crosstab(scoreAndLabelsPandas['label'], scoreAndLabelsPandas['prediction'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True,cmap='Blues', fmt='g')

plt.show()

# COMMAND ----------

true_positive = scoreAndLabelsPandas[(scoreAndLabelsPandas.label == 1) & (scoreAndLabelsPandas.prediction == 1) ]['label'].count()
true_negative = scoreAndLabelsPandas[(scoreAndLabelsPandas.label == 0) & (scoreAndLabelsPandas.prediction == 0)]['label'].count()
false_positive = scoreAndLabelsPandas[(scoreAndLabelsPandas.label == 0) & (scoreAndLabelsPandas.prediction == 1)]['prediction'].count()
false_negative = scoreAndLabelsPandas[(scoreAndLabelsPandas.label == 1) & (scoreAndLabelsPandas.prediction == 0)]['label'].count()

precision = true_positive / (false_positive + true_positive)
recall = true_positive / (false_negative + true_positive)
f1 = (2 * precision * recall)/ (precision + recall)

# COMMAND ----------

print('''Precision: {}
Recall: {}
F1: {}'''.format(precision, recall, f1))

# COMMAND ----------

precision = precision_score(scoreAndLabelsPandas.label.astype('int'), scoreAndLabelsPandas.prediction.astype('int'))
recall = recall_score(scoreAndLabelsPandas.label, scoreAndLabelsPandas.prediction)
f1 = f1_score(scoreAndLabelsPandas.label, scoreAndLabelsPandas.prediction)
accuracy = (true_positive + true_negative)/ (true_positive + true_negative + false_positive + false_negative)


print('''Precision: {}
Recall: {}
F1: {}
Accuracy: {}'''.format(precision, recall, f1, accuracy))

# COMMAND ----------

origin_aa1_rain_depth = train_set.select('origin_aa1_rain_depth')

origin_aa1_rain_depth = origin_aa1_rain_depth.rdd.flatMap(lambda x: x).histogram(11)

# Loading the Computed Histogram into a Pandas Dataframe for plotting
pd.DataFrame(
    list(zip(*origin_aa1_rain_depth)), 
    columns=['bin', 'frequency']
).set_index(
    'bin'
).plot(kind='bar');
