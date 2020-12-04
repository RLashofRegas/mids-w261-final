# Databricks notebook source
# MAGIC %md #### Create toy example of random forests for final notebook

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
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.mllib.tree import RandomForest, RandomForestModel

# COMMAND ----------

# initialize the sql context
sqlContext = SQLContext(sc)

# COMMAND ----------

# global variables

# shared directory for our team (make sure it exists)
final_project_path = "dbfs:/mnt/mids-w261/group_5/"
dbutils.fs.mkdirs(final_project_path)

# input data paths
weather_data_path = "dbfs:/mnt/mids-w261/datasets_final_project/weather_data/weather20*.parquet"
airlines_data_path = "dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/20*.parquet"
city_timezone_path = final_project_path + "city_timezones.csv"

# output paths
train_data_output_path = final_project_path + "training_data_output/train.parquet"
validation_data_output_path = final_project_path + "training_data_output/validation.parquet"
test_data_output_path = final_project_path + "training_data_output/test.parquet"
train_data_output_path_one_hot = final_project_path + "training_data_output/train_one_hot.parquet"
validation_data_output_path_one_hot = final_project_path + "training_data_output/validation_one_hot.parquet"
test_data_output_path_one_hot = final_project_path + "training_data_output/test_one_hot.parquet"

# COMMAND ----------

# Read in parquet file
train_set = spark.read.parquet(train_data_output_path)
val_set = spark.read.parquet(validation_data_output_path)
test_set = spark.read.parquet(test_data_output_path)

# COMMAND ----------

train_set.printSchema()

# COMMAND ----------

# MAGIC %md ## Algorithm Implementation

# COMMAND ----------

# MAGIC %md We have selected Random Forests (RF) as the final model based on results from the exploritory algorithm analysis. We will explain and demonstrate the RF algorithm using a select portion of the train and test data on flight delays. 

# COMMAND ----------

# MAGIC %md #### Toy Example

# COMMAND ----------

# MAGIC %md First we select a few features from the dataset to visualize the trees that RF will build and compile these into a feature vector for the model.

# COMMAND ----------

train_toy = train_set.select("label", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index", "origin_avg_dep_delay", "day_of_week_Index", "crs_dep_hour_Index")
val_toy = val_set.select("label", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index", "origin_avg_dep_delay", "day_of_week_Index", "crs_dep_hour_Index")
test_toy = test_set.select("label", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index", "origin_avg_dep_delay", "day_of_week_Index", "crs_dep_hour_Index")

# COMMAND ----------

# vector of features
features = ["PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index", "origin_avg_dep_delay", "day_of_week_Index", "crs_dep_hour_Index"]
assembler = VectorAssembler(inputCols=features, outputCol="features")

train_toy = assembler.transform(train_toy)
val_toy = assembler.transform(val_toy)
test_toy = assembler.transform(test_toy)

# COMMAND ----------

# MAGIC %md Next we will fit the model by building trees - each constructed with a series of splitting rules. In the model below, the first node and top of the tree is split based on whether the previous flight is delayed. From these branches the next split is on average departure delay at the origin airport. We continue building the tree in this manner. RF trees are built in a unique way such that each node is randomly assigned a subset of features that will be considered as possible split candidates. This guarantees that the trees will differ from each other, which when averaged will create a more reliable result. Continuously splitting creates regions, depending on the combinations of features, to which training examples are assigned.
# MAGIC 
# MAGIC How does the model decide these split points?  In the training phase of a classification tree, the splitting point is the point which minimizes the *gini index*, which is a measure of node purity.  
# MAGIC $$ G = \sum {\hat {p}} $$
# MAGIC 
# MAGIC In a classification tree, we assign a test data point to the leaf of the tree to which it belongs based on its features, and it is assigned to the majority class of that region. 

# COMMAND ----------

rf = RF(labelCol="label", featuresCol="features", numTrees=10)

# COMMAND ----------

RF_model = rf.fit(train_toy)

# COMMAND ----------

display(RF_model)

# COMMAND ----------

# MAGIC %md Many trees are used in RF to reduce variance and increase reliability.