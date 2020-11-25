# Databricks notebook source
# MAGIC %md
# MAGIC # Airline Delays
# MAGIC ## Data Processing Pipeline
# MAGIC This notebook contains our data processing pipeline. First we read in the raw data. Then we perform transformations, feature engineering, and feature selection. After this we join our various datasets to produce a new dataset where each record contains all necessary data to be passed into our models. Finally we save off various stages of this process for use in our models and EDA. As part of this final step we also perform a train/validation/test split.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialization
# MAGIC In this section we perform basic initialization to be used in the rest of the notebook. We import all required packages, declare global variables, initialize our spark context, and read in our raw data into dataframes.

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

# read in raw data
airlines = spark.read.option("header", "true").parquet(airlines_data_path) # full airline dataset
weather = spark.read.option("header", "true").parquet(weather_data_path) # full weather dataset
city_timezone = spark.read.option("header", "false").csv(city_timezone_path) # table that maps city -> timezone

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Transformations

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airlines

# COMMAND ----------

# clean up origin and destination city names for join to timezone data

def split_city_name(city_state):
  '''UDF to deal with cases where dual cities are labeled
  with a "/". Returns only first city '''

  city = city_state.split(',')[0]
  state = city_state.split(',')[1]
  shortened_city = city.split('/')[0]
  
  return shortened_city + ',' + state

#convert function to udf
split_city_name_udf = udf(split_city_name, StringType())

# add new columns to airlines dataset
airlines = airlines \
              .withColumn("SHORT_DEST_CITY_NAME", split_city_name_udf('DEST_CITY_NAME')) \
              .withColumn("SHORT_ORIG_CITY_NAME", split_city_name_udf('ORIGIN_CITY_NAME'))

# COMMAND ----------

# Create SQL view of reference table for use in UTC conversion below
city_timezone.createOrReplaceTempView("city_timezone")
sqlContext.sql("""
DROP VIEW IF EXISTS city_state_timezone
""")
# convert to friendly column names and concatenate city and state for join
sqlContext.sql("""
CREATE TEMPORARY VIEW city_state_timezone
AS
SELECT 
  _c0 AS city_id,
  _c1 AS city,
  _c2 AS country,
  _c3 AS state,
  _c4 AS timezone,
  CONCAT(_c1, ', ', _c3) AS city_state
FROM city_timezone
""")

# COMMAND ----------

# Filter on selected fields & add timezones
# Time zone conversion:
#     1. pad time with zeros to make all times of the form HHmm
#     2. concatenate with flight date
#     3. convert to timestamp (local time)
#     4. truncate datetime to the hour
#     5. convert to UTC time using city_timezones dataset
airlines.createOrReplaceTempView("airlines_temp")
sqlContext.sql("""
DROP VIEW IF EXISTS airlines
""")
sqlContext.sql("""
CREATE TEMPORARY VIEW airlines
AS
SELECT
  year,
  quarter,
  month,
  day_of_week,
  fl_date,
  op_unique_carrier,
  tail_num,
  origin_airport_id,
  origin,
  origin_city_name,
  dest_airport_id,
  dest,
  dest_city_name,
  crs_dep_time,
  dep_time,
  dep_delay,
  dep_del15,
  cancelled,
  diverted,
  short_dest_city_name,
  short_orig_city_name,
  carrier_delay,
  weather_delay,
  nas_delay,
  security_delay,
  late_aircraft_delay,
  taxi_out,
  td.timezone AS dest_timezone,
  to.timezone AS origin_timezone,
  TO_UTC_TIMESTAMP(DATE_TRUNC('hour', TO_TIMESTAMP(CONCAT(fl_date, ' ', LPAD(crs_dep_time, 4, '0')), 'yyyy-MM-dd HHmm')), to.timezone) AS truncated_crs_dep_time_utc,
  TO_UTC_TIMESTAMP(DATE_TRUNC('hour', TO_TIMESTAMP(CONCAT(fl_date, ' ', LPAD(crs_dep_time, 4, '0')), 'yyyy-MM-dd HHmm')), to.timezone) - INTERVAL 3 HOURS AS truncated_crs_dep_minus_three_utc,
  TO_UTC_TIMESTAMP(TO_TIMESTAMP(CONCAT(fl_date, ' ', LPAD(crs_dep_time, 4, '0')), 'yyyy-MM-dd HHmm'), to.timezone) AS crs_dep_time_utc,
  TO_UTC_TIMESTAMP(TO_TIMESTAMP(CONCAT(fl_date, ' ', LPAD(crs_dep_time, 4, '0')), 'yyyy-MM-dd HHmm'), to.timezone) - INTERVAL 2 HOURS 15 MINUTES AS crs_dep_minus_two_fifteen_utc
FROM airlines_temp AS f
LEFT JOIN city_state_timezone AS td ON
  f.short_dest_city_name = td.city_state
LEFT JOIN city_state_timezone AS to ON
  f.short_orig_city_name = to.city_state
""")

# make sure view is cached for subsequent operations
sqlContext.cacheTable("airlines")

# reassign airlines df to derived view and cache it
airlines = sqlContext.sql("SELECT * FROM airlines").cache()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather

# COMMAND ----------

# add airport code to weather data to facilitate join
# filter to only records with valid airport code

def create_airport_code_stations(call_code):
  """
  This method creates an airport code from the call sign in the weather data.
  Call signs that start with 'K' correspond to weather stations at airports and match airport codes.
  If the call sign either does not start with K or is less than 5 characters the airport code will be blank.
  """
  try:
    if call_code[0] == 'K':
      airport_code = call_code[1:4]
    else:
      airport_code = ''
  except:
    airport_code = ''
  return airport_code

# convert function to udf
create_airport_code_stations_udf = udf(create_airport_code_stations, types.StringType())
# add airport code to weather dataset
weather = weather.withColumn("airport_code", create_airport_code_stations_udf('CALL_SIGN'))
# filter weather to records with valid airport code
weather = weather.where(col("airport_code") != '')

# COMMAND ----------

# Truncate weather data to hour
# For each hour take weather reading closest to hour per airport code
weather.createOrReplaceTempView("weather_temp")
sqlContext.sql("""
DROP VIEW IF EXISTS weather
""")
sqlContext.sql("""
CREATE TEMPORARY VIEW weather
AS
WITH weather_with_hour
AS
(
  SELECT
    wt.*,
    DATE_TRUNC('hour', TO_TIMESTAMP(wt.date, "yyyy-MM-ddTHH:mm:ss 'UTC'")) AS hour
  FROM weather_temp AS wt
),
weather_ranked
AS
(
  SELECT
    wh.*,
    ROW_NUMBER() OVER(PARTITION BY wh.hour, wh.airport_code ORDER BY wh.date) as rank
  FROM weather_with_hour AS wh
)
SELECT
  wr.date,
  wr.name,
  wr.report_type,
  wr.quality_control,
  wr.wnd,
  wr.cig,
  wr.vis,
  wr.tmp,
  wr.dew,
  wr.slp,
  wr.airport_code,
  wr.hour,
  wr.aw1
FROM weather_ranked AS wr
WHERE
  wr.rank = 1
""")

# make sure view is cached for subsequent operations
sqlContext.cacheTable("weather")

# reassign weather df to derived view and cache it
weather = sqlContext.sql("SELECT * FROM weather").cache()

# COMMAND ----------

#Source: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf

#Functions to Mandatory Weather Data - Parse Weather Variables WND - WIND-OBSERVATION, CIG - SKY-CONDITION-OBSERVATION, VIS - VISIBILITY-OBSERVATION, TMP - AIR-TEMPERATURE-OBSERVATION,  DEW - DEW POINT, SLP = Sea Level AIR-PRESSURE-OBSERVATION
def wind_parse(df,column_name='WND'):
  split_col = f.split(df[column_name], ',')
  
  #direction angle  999 = Missing. If type code (below) = V, then 999 indicates variable wind direction.
  direction_angle_udf = udf(lambda x: None if x == "999" else x,IntegerType())
  df = df.withColumn(column_name + '_direction_angle', direction_angle_udf(split_col.getItem(0)))
  
  df = df.withColumn(column_name + '_direction_quality', split_col.getItem(1))
  
  #WIND-OBSERVATION type code  NOTE: If a value of 9 appears with a wind speed of 0000, this indicates calm winds.
  wind_type_udf = udf(lambda x: None if x == "9" else x)
  df = df.withColumn(column_name + '_type_code', wind_type_udf(split_col.getItem(2)))
  
  #speed rate 9999 = Missing, fix formatting to be integer MIN: 0000 MAX: 0900
  speed_udf = udf(lambda x: None if x == "9999" else x,IntegerType())
  df = df.withColumn(column_name + '_speed_rate', speed_udf(split_col.getItem(3))) #Likely most important code
  
  #WIND-OBSERVATION type code  NOTE: If a value of 9 appears with a wind speed of 0000, this indicates calm winds.
  #df = df.withColumn(column_name + '_type_code', f.when(df[column_name + '_speed_rate'] == '0000' & df[column_name + '_type_code'] == 'null', "Calm")
  
  ##direction angle  999 = Missing. If type code (below) = V, then 999 indicates variable wind direction.
  ###

  df = df.withColumn(column_name + '_speed__quality', split_col.getItem(4))
  return df

def sky_parse(df,column_name='CIG'):
  split_col = f.split(df[column_name], ',')
  
  ceiling_height_udf = udf(lambda x: None if x == "99999" else x,IntegerType())
  df = df.withColumn(column_name + '_ceiling_height', ceiling_height_udf(split_col.getItem(0)))
  
  df = df.withColumn(column_name + '_ceiling_quality', split_col.getItem(1))
  
  ceiling_det_vis_udf = udf(lambda x: None if x == "9" else x)
  df = df.withColumn(column_name + '_ceiling_determination', ceiling_det_vis_udf(split_col.getItem(2)))
  df = df.withColumn(column_name + '_ceiling_visibility_okay', ceiling_det_vis_udf(split_col.getItem(3))) #Likely most important code
  
  return df

def visibility_parse(df,column_name='VIS'):
  split_col = f.split(df[column_name], ',')
  
  vis_distance_udf = udf(lambda x: None if x == "999999" else x,IntegerType())
  df = df.withColumn(column_name + '_distance', vis_distance_udf(split_col.getItem(0))) #Likely most important code
  
  
  df = df.withColumn(column_name + '_distance_quality', split_col.getItem(1))
  
  vis_variability_udf = udf(lambda x: None if x == "9" else x)
  df = df.withColumn(column_name + '_variability', vis_variability_udf(split_col.getItem(2)))
  
  df = df.withColumn(column_name + '_quality_variability', split_col.getItem(3)) 
  return df

def tmp_parse(df,column_name='TMP'):
  split_col = f.split(df[column_name], ',')
  
  air_temp_udf = udf(lambda x: None if x == "+9999" else x,IntegerType())
  df = df.withColumn(column_name + '_air_temperature', air_temp_udf(split_col.getItem(0))) #Likely most important code
  
  df = df.withColumn(column_name + '_air_temperature_quality', split_col.getItem(1))
  return df

def dew_parse(df,column_name='DEW'):
  split_col = f.split(df[column_name], ',')
  
  dew_temp_udf = udf(lambda x: None if x == "+9999" else x,IntegerType())
  df = df.withColumn(column_name + '_dew_point_temp', dew_temp_udf(split_col.getItem(0))) #Likely most important code
  
  df = df.withColumn(column_name + '_dew_point_temp_quality', split_col.getItem(1))
  return df

def slp_parse(df,column_name='SLP'):
  split_col = f.split(df[column_name], ',')
  
  slp_udf = udf(lambda x: None if x == "99999" else x,IntegerType())
  df = df.withColumn(column_name + '_sea_level_pressure', slp_udf(split_col.getItem(0))) #Likely most important code, low-pressure system moves into an area, it usually leads to cloudiness, wind, and precipitation
  
  df = df.withColumn(column_name + '_sea_level_pressure_quality', split_col.getItem(1))
  return df


# Additional Weather Data 

# Present weather indicator  - need to get 2 hours before flight - blank if no data p30 for descriptions
# Automated_atmospheric_condition codes are used to report precipitation, fog, thunderstorm at the station during the preceding hour, but not at the time of observation.)
def present_weather_parse(df,column_name='AW1'):
#When string is empty put in a filler to enable parsing
  blank_string_udf = udf(lambda x: "," if x == "" else x, StringType())
  df = df.withColumn("AW1_New", blank_string_udf(df[column_name]))
  split_col = f.split(df["AW1_New"], ',')
  
#Replace missing data with nulls   
  present_udf = udf(lambda x: None if x == "" else x)
  df = df.withColumn(column_name + '_automated_atmospheric_condition', present_udf(split_col.getItem(0))) #Likely most important code
  df = df.withColumn(column_name + '_quality_automated_atmospheric_condition', present_udf(split_col.getItem(1)))
  
  return df

weather = wind_parse(weather)  
weather = sky_parse(weather)  
weather = visibility_parse(weather)  
weather = tmp_parse(weather)  
weather = dew_parse(weather)  
weather = slp_parse(weather)  
weather = present_weather_parse(weather)
weather = weather.cache()

weather.createOrReplaceTempView("weather")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering
# MAGIC In this section we take the basic parsed versions of the weather and airlines data above and perform some feature engineering to add a few columns we think will be useful. We focus on two categories of delays - chain delays and root cause delays. The `delays_by_airport`, `delays_by_carrier` (also grouped by airport), and `Holiday` features attempt to capture root cause delays. The `chain_delay_feature_engineering` function joins on `tail_num` in order to capture previous delays for the same physical airplane on a given day (chain delays).

# COMMAND ----------

# Aggregate delays by airport and hour
sqlContext.sql("""
DROP VIEW IF EXISTS delays_by_airport
""")
sqlContext.sql("""
CREATE TEMPORARY VIEW delays_by_airport
AS
SELECT
  a.origin,
  a.truncated_crs_dep_time_utc AS hour,
  COUNT(*) AS num_flights,
  AVG(dep_delay) AS avg_dep_delay,
  AVG(dep_del15) AS pct_dep_del15,
  AVG(taxi_out) AS avg_taxi_time,
  AVG(weather_delay) AS avg_weather_delay,
  AVG(nas_delay) AS avg_nas_delay,
  AVG(security_delay) AS avg_security_delay,
  AVG(late_aircraft_delay) AS avg_late_aircraft_delay
FROM airlines AS a
GROUP BY
  a.origin,
  a.truncated_crs_dep_time_utc
""")

# COMMAND ----------

# Aggregate delays by airport, carrier, and hour
sqlContext.sql("""
DROP VIEW IF EXISTS delays_by_carrier
""")
sqlContext.sql("""
CREATE TEMPORARY VIEW delays_by_carrier
AS
SELECT
  a.origin,
  a.op_unique_carrier,
  a.truncated_crs_dep_time_utc AS hour,
  COUNT(*) AS num_flights,
  AVG(dep_delay) AS avg_dep_delay,
  AVG(carrier_delay) AS avg_carrier_delay
FROM airlines AS a
GROUP BY
  a.origin,
  a.truncated_crs_dep_time_utc,
  a.op_unique_carrier
""")

# COMMAND ----------

def holiday_column(airline_df, start='2014-01-01', end='2018-12-31'):
  '''Takes airline df and returns df with column indicating if
  date of flight is a US Federal Holiday'''

  #Pull Holiday Dates and convert to Spark DF with timestamp column
  cal = USFederalHolidayCalendar()
  holidays = cal.holidays(start, end).to_pydatetime()
  holidays_df = pd.DataFrame(pd.DataFrame(holidays)[0].astype('string'))
  schema = StructType([StructField('Holiday_Date', StringType())])
  #holidays_sc = sc.parallelize(pd.DataFrame(holidays_df)).toDF(['Holiday_Date'])
  holidays_sc = spark.createDataFrame(holidays_df, schema)
  holidays_sc = holidays_sc.select(to_timestamp(holidays_sc.Holiday_Date, 'yyyy-MM-dd').alias('holiday_date'))
  
  #Join holidays to airlines
  holiday_joined_df = airline_df.join(holidays_sc, 
                              (airline_df.fl_date == holidays_sc.holiday_date),
                              'left')

  #Change date column to binary
  holiday_joined_df = holiday_joined_df.withColumn("Holiday", (f.col('holiday_date').isNotNull()).cast("integer"))
  
  #Drop redundant holiday_date column
  holiday_joined_df = holiday_joined_df.drop(holiday_joined_df.holiday_date)
  
  return holiday_joined_df

# add is_holiday column to airlines dataset
airlines = holiday_column(airlines, end='2020-01-01')

# replace temp view
airlines.createOrReplaceTempView("airlines")

# COMMAND ----------

def chain_delay_feature_engineering(airline_df):
  '''Takes airline df with created columns CRS_DEP_TIME_UTC, CRS_DEP_MINUS_TWO_FIFTEEN_UTC and returns new airline df with 5 added columns:
  dep_time_diff_one_flight_before: time between departure of current flight and previous flight (in seconds)
  dep_time_diff_two_flights_before: time between departure of current flight and  flight two previous (in seconds)
  delay_one_before: was flight before delayed? (binary)
  delay_two_before: was flight two before delayed? (binary)
  PREVIOUS_FLIGHT_DELAYED_FOR_MODEL: If previous flight is at least 2 hours 15 minutes prior (8100 seconds), was it delayed? If less than 2:15, was flight 2 before delayed? (binary)'''
     
  airline_df.createOrReplaceTempView("airlines_temp_view")
  #Filter out cancelled flights
  airlines_non_cancelled = sqlContext.sql("""
  SELECT *
  FROM airlines_temp_view WHERE CANCELLED != 1
  """)
  
  #Store new df with limited number of ordered columns that we can use to window 
  airlines_aircraft_tracking = airlines_non_cancelled[["tail_num","fl_date","origin_city_name", "dest_city_name", "dep_del15", "crs_dep_time_utc", "crs_dep_minus_two_fifteen_utc"]].orderBy("tail_num","fl_date", "crs_dep_time_utc")
  
  #This section is related to windowing so that we can pull information from previous flight and flight 2 before current flight. Windowing will only pull for the same tail number
  w = Window.partitionBy("tail_num").orderBy("crs_dep_time_utc")

  diff = col("crs_dep_time_utc").cast("long") - lag("crs_dep_time_utc", 1).over(w).cast("long")
  diff2 = col("crs_dep_time_utc").cast("long") - lag("crs_dep_time_utc", 2).over(w).cast("long")
  delay_one_before = lag("dep_del15", 1).over(w)
  delay_two_before = lag("dep_del15", 2).over(w)

  airlines_aircraft_tracking_diff = airlines_aircraft_tracking.withColumn("dep_time_diff_one_flight_before", diff)\
                                  .withColumn("dep_time_diff_two_flights_before", diff2)\
                                  .withColumn("delay_one_before", delay_one_before)\
                                  .withColumn("delay_two_before", delay_two_before)

  def chain_delay_analysis (dep_time_diff_one_flight_before, dep_time_diff_two_flights_before,
                           delay_one_before, delay_two_before):
    '''Takes info on flight before: departure time difference, whether
    flight was delayed and returns 1 if flight before was delayed AND outside of 2:15 from current flight.
    If outside of 2:15 looks at flight 2 before and returns 1 if that one was delayed, 0 if not.'''
    try:
      if dep_time_diff_one_flight_before >= 8100:
        return int(delay_one_before)
      else:
        return int(delay_two_before)
    except:
      return 'null'
  #NEED TO RETURN TO THIS AND DECIDE WHAT TO DO IF FLIGHT 2 PREVIOUS WAS WITHIN 2:15. I BELIEVE MOST (ALL?) OF THOSE ARE DATA CLEANSING ISSUES
  #ALSO SHOULD LOOK AT THE CASES WHERE THE PREVIOUS FLIGHT WAS SO MUCH EARLIER THAT ITS DELAY DOES NOT REALLY AFFECT CURRENT FLIGHT. HOW MUCH EARLIER
  #SHOULD THAT BE?

  chain_delay_analysis_udf = f.udf(chain_delay_analysis)

  airlines_aircraft_tracking_diff_for_join = airlines_aircraft_tracking_diff.withColumn("PREVIOUS_FLIGHT_DELAYED_FOR_MODELS", chain_delay_analysis_udf('dep_time_diff_one_flight_before', 'dep_time_diff_two_flights_before',
                           'delay_one_before', 'delay_two_before'))
  
  airline_df_with_id = airline_df.withColumn("id", monotonically_increasing_id())
  
  join_columns = ["tail_num","fl_date","origin_city_name", "dest_city_name", "crs_dep_time_utc"]
  
  airlines_chain_delays = airline_df_with_id.alias("a").join(airlines_aircraft_tracking_diff_for_join.alias("j"), join_columns, 'left_outer') \
                            .select('a.year', 'a.quarter', 'a.month', 'a.day_of_week', 'a.fl_date', 'a.op_unique_carrier', 'a.tail_num', 'a.origin_airport_id', 'a.origin', 'a.origin_city_name', 'a.dest_airport_id', 'a.dest', 'a.dest_city_name', 'a.crs_dep_time', 'a.dep_time', 'a.dep_delay', 'a.dep_del15', 'a.cancelled', 'a.diverted', 'a.short_dest_city_name', 'a.short_orig_city_name', 'a.carrier_delay', 'a.weather_delay', 'a.nas_delay', 'a.security_delay', 'a.late_aircraft_delay', 'a.taxi_out', 'a.dest_timezone', 'a.origin_timezone', 'a.truncated_crs_dep_time_utc', 'a.truncated_crs_dep_minus_three_utc', 'a.crs_dep_time_utc', 'a.crs_dep_minus_two_fifteen_utc', 'a.Holiday', 'a.id', 'j.dep_time_diff_one_flight_before', 'j.dep_time_diff_two_flights_before', 'j.delay_one_before', 'j.delay_two_before', 'j.PREVIOUS_FLIGHT_DELAYED_FOR_MODELS')
  
  #Drop duplicates created during join. Is there a better way?
  airlines_chain_delays_no_dups = airlines_chain_delays.dropDuplicates(['id'])
  
  return airlines_chain_delays_no_dups

# add chain delay features
airlines = chain_delay_feature_engineering(airlines)

# replace temp view
airlines.createOrReplaceTempView("airlines")

# COMMAND ----------

# Bin departure times
## splits = [400,800,1200,1600,2000,2400]
## bucketizer = Bucketizer(splits=splits, inputCol="crs_dep_time", outputCol="CRS_DEP_TIME_Daypart")
## airlines = bucketizer.transform(airlines).cache()

# replace temp view
## airlines.createOrReplaceTempView("airlines")

# COMMAND ----------

def crs_dep_hour(crs_dep_time):
  '''
  Takes crs_dep_time scheduled departure time and creates categorical variable for hour of scheduled departure.
  '''
  str_time = str(crs_dep_time)
  if len(str_time) < 4:
      crs_dep_hour = str(0) + str_time[0]
  elif len(str_time) == 4:
      crs_dep_hour = str_time[:2]
  return crs_dep_hour
  
crs_dep_hour_udf = f.udf(crs_dep_hour, StringType())
airlines = airlines.withColumn("crs_dep_hour", crs_dep_hour_udf("crs_dep_time"))

# replace temp view
airlines.createOrReplaceTempView("airlines")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join Airlines to Weather Data
# MAGIC In this section we do the final join between airlines and weather data (for both origin and destination airports). We make sure to only join each flight to weather data from at least 2 hours before the scheduled departure of the flight.

# COMMAND ----------

#JOIN WEATHER AND AIRLINE DATA FOR ORIGIN BETWEEN 2-3 hours DEFORE DEPARTURE
weather_airline_joined = sqlContext.sql("""
SELECT
  f.*,
  dao.num_flights AS origin_num_flights,
  dao.avg_dep_delay AS origin_avg_dep_delay,
  dao.pct_dep_del15 AS origin_pct_dep_del15,
  dao.avg_taxi_time AS origin_avg_taxi_time,
  dao.avg_weather_delay AS origin_avg_weather_delay,
  dao.avg_nas_delay AS origin_avg_nas_delay,
  dao.avg_security_delay AS origin_avg_security_delay,
  dao.avg_late_aircraft_delay AS origin_avg_late_aircraft_delay,
  dad.num_flights AS dest_num_flights,
  dad.avg_dep_delay AS dest_avg_dep_delay,
  dad.pct_dep_del15 AS dest_pct_dep_del15,
  dad.avg_taxi_time AS dest_avg_taxi_time,
  dad.avg_weather_delay AS dest_avg_weather_delay,
  dad.avg_nas_delay AS dest_avg_nas_delay,
  dad.avg_security_delay AS dest_avg_security_delay,
  dad.avg_late_aircraft_delay AS dest_avg_late_aircraft_delay,
  dco.num_flights AS carrier_num_flights,
  dco.avg_dep_delay AS carrier_avg_dep_delay,
  dco.avg_carrier_delay AS carrier_avg_carrier_delay,
  wo.WND_direction_quality AS origin_WND_direction_quality,
  wo.WND_type_code AS origin_WND_type_code,
  wo.WND_speed_rate AS origin_WND_speed_rate,
  wo.WND_speed__quality AS origin_WND_speed__quality,
  wo.CIG_ceiling_height AS origin_CIG_ceiling_height,
  wo.CIG_ceiling_quality AS origin_CIG_ceiling_quality,
  wo.CIG_ceiling_visibility_okay AS origin_CIG_ceiling_visibility_okay,
  wo.VIS_distance AS origin_VIS_distance,
  wo.VIS_distance_quality AS origin_VIS_distance_quality,
  wo.VIS_variability AS origin_VIS_variability,
  wo.VIS_quality_variability AS origin_VIS_quality_variability,
  wo.TMP_air_temperature AS origin_TMP_air_temperature,
  wo.TMP_air_temperature_quality AS origin_TMP_air_temperature_quality,
  wo.DEW_dew_point_temp AS origin_DEW_dew_point_temp,
  wo.DEW_dew_point_temp_quality AS origin_DEW_dew_point_temp_quality,
  wo.SLP_sea_level_pressure_quality AS origin_SLP_sea_level_pressure_quality,
  wo.aw1_automated_atmospheric_condition AS origin_aw1_automated_atmospheric_condition,
  wo.aw1_quality_automated_atmospheric_condition AS origin_aw1_quality_automated_atmospheric_condition,
  wd.WND_direction_quality AS dest_WND_direction_quality,
  wd.WND_type_code AS dest_WND_type_code,
  wd.WND_speed_rate AS dest_WND_speed_rate,
  wd.WND_speed__quality AS dest_WND_speed__quality,
  wd.CIG_ceiling_height AS dest_CIG_ceiling_height,
  wd.CIG_ceiling_quality AS dest_CIG_ceiling_quality,
  wd.CIG_ceiling_visibility_okay AS dest_CIG_ceiling_visibility_okay,
  wd.VIS_distance AS dest_VIS_distance,
  wd.VIS_distance_quality AS dest_VIS_distance_quality,
  wd.VIS_variability AS dest_VIS_variability,
  wd.VIS_quality_variability AS dest_VIS_quality_variability,
  wd.TMP_air_temperature AS dest_TMP_air_temperature,
  wd.TMP_air_temperature_quality AS dest_TMP_air_temperature_quality,
  wd.DEW_dew_point_temp AS dest_DEW_dew_point_temp,
  wd.DEW_dew_point_temp_quality AS dest_DEW_dew_point_temp_quality,
  wd.SLP_sea_level_pressure_quality AS dest_SLP_sea_level_pressure_quality,
  wd.aw1_automated_atmospheric_condition AS dest_aw1_automated_atmospheric_condition,
  wd.aw1_quality_automated_atmospheric_condition AS dest_aw1_quality_automated_atmospheric_condition
FROM airlines AS f
LEFT JOIN weather AS wo ON
  f.origin = wo.airport_code
  AND f.truncated_crs_dep_minus_three_utc = wo.hour
LEFT JOIN weather AS wd ON
  f.dest = wd.airport_code
  AND f.truncated_crs_dep_minus_three_utc = wd.hour
LEFT JOIN delays_by_airport AS dao ON
  f.origin = dao.origin
  AND f.truncated_crs_dep_minus_three_utc = dao.hour
LEFT JOIN delays_by_airport AS dad ON
  f.dest = dad.origin
  AND f.truncated_crs_dep_minus_three_utc = dad.hour
LEFT JOIN delays_by_carrier AS dco ON
  f.origin = dco.origin
  AND f.truncated_crs_dep_minus_three_utc = dco.hour
  AND f.op_unique_carrier = dco.op_unique_carrier
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train/Test Split and Save Data
# MAGIC In this section we take the joined data from above and split it into train, validation, and test sets. For this dataset we choose to perform our train-test split using the years of data (2015-17 for train, 2018 for validation, and 2019 for test). This choice has two primary benefits:
# MAGIC 1. We maintain relational data elements between the flights. This includes flights for the same day for calculating aggregated data by airport and chain delays for a given tail number for our feature engineering, as well as time-based patterns such as yearly seasonality, holidays, and flights over the course of the day.
# MAGIC 2. This aligns with a real world business process. In a real world scenario we would be using data from previous years to predict the next year.
# MAGIC 
# MAGIC The downside of this approach is that we may miss or over fit to features that are related to a specific year, but we believe this is secondary to the benefits stated above.
# MAGIC 
# MAGIC After our data is split we save the output files for later use in building our models.

# COMMAND ----------

# cached join data to reduce processing
cached_join = weather_airline_joined.cache()

# perform train/test split based on year
train_set = cached_join.where((col("year") == '2015') | (col("year") == '2016') | (col("year") == '2017')).cache()
val_set = cached_join.where((col("year") == '2018')).cache()
test_set = cached_join.where((col("year") == '2019')).cache()

train_set.write.format("parquet").mode("overwrite").save(train_data_output_path)
val_set.write.format("parquet").mode("overwrite").save(validation_data_output_path)
test_set.write.format("parquet").mode("overwrite").save(test_data_output_path)

# COMMAND ----------

# Index label
labelIndexer = StringIndexer(inputCol="dep_del15", outputCol="label")
labelIndexer.fit(train_GBT)
train_GBT = labelIndexer.transform(train_GBT)
validation_GBT = labelIndexer.transform(validation_GBT)
test_GBT = labelIndexer.transform(test_GBT)

# Index features
categorical = ["month", "day_of_week", "op_unique_carrier", "Holiday", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS", "origin_WND_type_code", "origin_CIG_ceiling_visibility_okay", "origin_VIS_variability", "dest_WND_type_code", "dest_CIG_ceiling_visibility_okay", "dest_VIS_variability", "crs_dep_hour"]

categorical_index = [i + "_Index" for i in categorical]
  
stringIndexer = StringIndexer(inputCols=categorical, outputCols=categorical_index)
    stringIndexer.fit(train_GBT)
train_GBT = stringIndexer.transform(train_GBT)
validation_GBT = stringIndexer.transform(validation_GBT)
test_GBT = stringIndexer.transform(test_GBT)

# COMMAND ----------

# Assemble categorical and numeric features into vector
numeric = ["origin_num_flights","origin_avg_dep_delay", "origin_pct_dep_del15", "origin_avg_taxi_time", "origin_avg_weather_delay", "origin_avg_nas_delay", "origin_avg_security_delay", "origin_avg_late_aircraft_delay", "dest_num_flights","dest_avg_dep_delay", "dest_pct_dep_del15", "dest_avg_taxi_time", "dest_avg_weather_delay", "dest_avg_nas_delay", "dest_avg_security_delay", "dest_avg_late_aircraft_delay", "carrier_num_flights", "carrier_avg_dep_delay", "carrier_avg_carrier_delay", "origin_WND_speed_rate", "origin_CIG_ceiling_height", "origin_VIS_distance", "origin_TMP_air_temperature", "origin_DEW_dew_point_temp", "dest_WND_speed_rate", "dest_CIG_ceiling_height", "dest_VIS_distance", "dest_TMP_air_temperature", "dest_DEW_dew_point_temp"]

features = categorical_index + numeric
assembler = VectorAssembler(inputCols=features, outputCol="features")

train_GBT = assembler.transform(train_GBT)
validation_GBT = assembler.transform(validation_GBT)
test_GBT = assembler.transform(test_GBT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### One Hot Encode Features And Save Copy
# MAGIC Some models like decision trees do better with/can handle the raw features without one-hot encoding applied. Thus we save two copies of our training data (one before one-hot encoding and one after). When performing the one-hot encoding we make sure to only use the train dataset to fit our encoder so that we are not "cheating" and bringing in data from the test or validation set.

# COMMAND ----------

#Select string variables for formatting

#Code to convert strings into indexer for one hot encoding
strings = [i for i in ['op_unique_carrier','origin_WND_direction_quality','origin_WND_type_code','origin_WND_speed__quality',
'origin_CIG_ceiling_quality','origin_VIS_distance_quality','origin_VIS_variability','origin_VIS_quality_variability','origin_TMP_air_temperature_quality','origin_DEW_dew_point_temp_quality','origin_SLP_sea_level_pressure_quality','dest_WND_direction_quality','dest_WND_type_code','dest_WND_speed__quality',
'dest_CIG_ceiling_quality','dest_VIS_distance_quality','dest_VIS_variability','dest_VIS_quality_variability','dest_TMP_air_temperature_quality','dest_DEW_dew_point_temp_quality','dest_SLP_sea_level_pressure_quality']]

# only 1 distinct value: ,'origin_DEW_dew_point_temp','dest_DEW_dew_point_temp'

strings_indexer = []

for i in strings:
  strings_indexer.append(i+"_Indexer")
  
indexer = StringIndexer(inputCols=strings, outputCols=strings_indexer)
model = indexer.fit(train_set)

train_one_hot = model.transform(train_set)
val_one_hot = model.transform(val_set)
test_one_hot = model.transform(test_set)

# COMMAND ----------

#Code to one hot encode categorical variables including bucketed data 

#Numeric non-continuous variables that will be one hot encoded 
numeric_columns_non_cont = ["month","day_of_week","origin_airport_id"] # ,"CRS_DEP_TIME_Daypart"

#Combine numeric variables with indexed strings for one hot encoding 
categorical_var = strings_indexer + numeric_columns_non_cont

list_of_categorical = [i for i in categorical_var]

list_encoders = []

for i in list_of_categorical:
  list_encoders.append(i+"_Indicator")
  
encoder = OneHotEncoder(inputCols=[i for i in list_of_categorical],
                        outputCols=[i for i in list_encoders])

model = encoder.fit(train_one_hot)

train_one_hot = model.transform(train_one_hot)
val_one_hot = model.transform(val_one_hot)
test_one_hot = model.transform(test_one_hot)

# COMMAND ----------

train_set.write.format("parquet").mode("overwrite").save(train_data_output_path_one_hot)
val_set.write.format("parquet").mode("overwrite").save(validation_data_output_path_one_hot)
test_set.write.format("parquet").mode("overwrite").save(test_data_output_path_one_hot)
