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

# output paths
train_data_output_path = final_project_path + "training_data_output/train.parquet"
test_data_output_path = final_project_path + "training_data_output/test.parquet"
train_toy_output_path = final_project_path + "training_data_output/train_toy.parquet"
test_toy_output_path = final_project_path + "training_data_output/test_toy.parquet"

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/group_5"))

# COMMAND ----------

# Read in parquet file
train_set = spark.read.parquet(train_data_output_path)
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

# MAGIC %md ##### Build Example 
# MAGIC First we select a few features from the dataset to visualize the trees that RF will build and compile these into a feature vector for the model.

# COMMAND ----------

train_toy, test_toy = train_set.select("label", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index", "origin_avg_dep_delay", "day_of_week_Index", "crs_dep_hour_Index").randomSplit([0.8, 0.2], seed = 1)

# COMMAND ----------

# vector of features
features = ["PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index", "origin_avg_dep_delay", "day_of_week_Index", "crs_dep_hour_Index"]
assembler = VectorAssembler(inputCols=features, outputCol="features")

train_toy = assembler.transform(train_toy)
test_toy = assembler.transform(test_toy)

# COMMAND ----------

train_toy.write.format("parquet").save()
test_toy.write.format("parquet").save()

# COMMAND ----------

# MAGIC %md ##### Fit the model   
# MAGIC Next we will fit the model by building decision trees with the training examples, each constructed with a series of splitting rules. In the model below, the first node and top of the tree is split based on whether the previous flight is delayed. From these branches the next split is on average departure delay at the origin airport. We continue adding splits to divide the training examples into regions based on the combination of their features. 
# MAGIC 
# MAGIC How does the model decide these split points?  In our classification tree, the splitting point is the point which minimizes the *gini index*, which is a measure of node purity. The equation for the gini index is shown below, where \\(\hat{p}\_{mk}\\) is the proportion of examples in region \\(m\\) of class \\(k\\). For flight delays we can see this will be minimized when \\(\hat{p}\_{m, delayed}\\) and \\(\hat{p}\_{m, not\;delayed}\\) is close to 0 or 1, which is when the examples are close to clearly divided between delayed and not delayed in region \\(m\\).  
# MAGIC 
# MAGIC $$ G = \sum\_{k=1}^{K} {\hat{p}\_{mk} (1 - \hat{p}\_{mk})} $$
# MAGIC 
# MAGIC The method of averaging many trees grown from different samples of the training data, or bagging, decreases variance of the model that would occur with any one tree. The RF training method goes a step further to help guarantee a more reliable result. RF trees are built such that each node is randomly assigned a subset of features that will be considered as possible split candidates. This means that the trees will differ from each other, which when averaged will decrease variance more than bagging alone.  

# COMMAND ----------

rf = RF(labelCol="label", featuresCol="features", numTrees=1)

# COMMAND ----------

RF_model = rf.fit(train_toy)

# COMMAND ----------

display(RF_model)

# COMMAND ----------

# MAGIC %md ##### Make predictions
# MAGIC For each tree, we assign a test data points to the region, or leaf, of the tree to which it belongs based on its features. The equation below for a classification tree shows that the predicted class for each test example is the majority class of the region to which it is assigned.  
# MAGIC \\(\hat{Y}\_{i,m}\\) is the prediction for a test example \\(i\\) placed in region \\(m\\) and \\(\hat{p}\_{m,k}\\) is the proportion of training examples in region \\(m\\) of class \\(k\\).    
# MAGIC $$
# MAGIC \begin{aligned}
# MAGIC \hat{Y}\_{i,m} &= \begin{cases}
# MAGIC 0 &\text{if } \hat{p}\_{m,0} \gt \hat{p}\_{m,1} \\\
# MAGIC 1 &\text{if } \hat{p}\_{m,0} \lt \hat{p}\_{m,1}
# MAGIC \end{cases}
# MAGIC \end{aligned}
# MAGIC $$
# MAGIC 
# MAGIC RF then combines these predictions for all trees using a majority vote.  
# MAGIC $$
# MAGIC \begin{aligned}
# MAGIC \hat{Y}\_{i,n} &= \begin{cases}
# MAGIC 0 &\text{if } \hat{p}\_{m,0} \gt \hat{p}\_{m,1} \\\
# MAGIC 1 &\text{if } \hat{p}\_{m,0} \lt \hat{p}\_{m,1}
# MAGIC \end{cases}
# MAGIC \end{aligned}
# MAGIC $$

# COMMAND ----------

predictions = 