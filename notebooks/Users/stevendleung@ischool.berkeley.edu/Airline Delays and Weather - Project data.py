# Databricks notebook source
# MAGIC %md # Airline delays 
# MAGIC ## Bureau of Transportation Statistics
# MAGIC https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236   
# MAGIC https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
# MAGIC 
# MAGIC 2015 - 2019

# COMMAND ----------

# MAGIC %md ### Additional sources
# MAGIC This might be useful in matching station codes to airports:
# MAGIC 1. http://dss.ucar.edu/datasets/ds353.4/inventories/station-list.html
# MAGIC 2. https://www.world-airport-codes.com/

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
from pyspark.sql import types
from pyspark.sql.functions import *
from datetime import datetime, timedelta

sqlContext = SQLContext(sc)


# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
final_project_path = userhome + "/FINAL_PROJECT/" 

# COMMAND ----------

FINAL_PROJECT_path_open = '/dbfs' + final_project_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(final_project_path)

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/weather_data"))

# COMMAND ----------

# MAGIC %md ### Pull airline data from parquet on dbfs

# COMMAND ----------

airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/201*.parquet")
display(airlines.sample(False, 0.00001))

# COMMAND ----------

airlines.printSchema()

# COMMAND ----------

f'{airlines.count():,}'

# COMMAND ----------

# MAGIC %md ### Pull timezone reference chart from csv on dbfs

# COMMAND ----------

city_timezone = spark.read.option("header", "false").csv("dbfs:/FileStore/tables/cities1000-4.csv")

# COMMAND ----------

# MAGIC %md # Weather and Join Work
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532

# COMMAND ----------

# MAGIC %md #### Pulling and filtering data

# COMMAND ----------

#All Weather Files
display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/weather_data"))

# COMMAND ----------

#All Weather Files
weather = spark.read.option("header", "true")\
                    .parquet(f"dbfs:/mnt/mids-w261/datasets_final_project/weather_data/*.parquet")

f'{weather.count():,}'

# COMMAND ----------

# WEATHER ONLY 2015
weather2015 = spark.read.option("header", "true")\
                    .parquet(f"dbfs:/mnt/mids-w261/datasets_final_project/weather_data/weather2015a.parquet")

f'{weather2015.count():,}'

# COMMAND ----------

def create_airport_code_stations(call_code):
    try:
      if call_code[0] == 'K':
        airport_code = call_code[1:4]
      else:
        airport_code = ''
    except:
      airport_code = ''
    return airport_code

create_airport_code_stations_udf = f.udf(create_airport_code_stations, types.StringType())
weather_q12015_coded = weather_q12015.withColumn("airport_code", create_airport_code_stations_udf('CALL_SIGN')) 

# COMMAND ----------

# MAGIC %md #### FILTER WEATHER TO Q1 2015. SAVE FILE TO DBFS

# COMMAND ----------

#FILTER on Q1 and Airport stations
weather_atl_ord_q12015 = weather2015.filter((weather2015_atl_ord.DATE <= "2015-03-31") & 
                            ((weather2015.NAME == "CHICAGO OHARE INTERNATIONAL AIRPORT, IL US") | \
                            (weather2015.NAME == "ATLANTA HARTSFIELD INTERNATIONAL AIRPORT, GA US")))
#WILL WANT TO TAKE OUT THE AIRPORT FILTER LATER SINCE WE NEED TO BRING IN WEATHER AT DESTINATION STATIONS AS WELL

# COMMAND ----------

#SAVE FILE TO DBFS
weather_atl_ord_q12015.write.format("parquet").save(final_project_path + "weather_atl_ord_q1_2015.parquet")

# COMMAND ----------

#FILTER on Q1
weather_q12015 = weather2015.filter(weather2015.DATE <= "2015-03-31")

# COMMAND ----------

# MAGIC %md ##### Add Airport code to weather data

# COMMAND ----------

def create_airport_code_stations(call_code):
    try:
      if call_code[0] == 'K':
        airport_code = call_code[1:4]
      else:
        airport_code = ''
    except:
      airport_code = ''
    return airport_code

create_airport_code_stations_udf = f.udf(create_airport_code_stations, types.StringType())
weather_q12015_coded = weather_q12015.withColumn("airport_code", create_airport_code_stations_udf('CALL_SIGN')) 

# COMMAND ----------

#SAVE FILE TO DBFS
weather_q12015_coded.write.format("parquet").save(final_project_path + "weather_q12015_coded.parquet")

# COMMAND ----------

# MAGIC %md #### FILTER FLIGHTS TO Q1 2015, DEPARTING ORD AND ATL. SAVE FILE TO DBFS

# COMMAND ----------

#FILTER on Q1 and Airport stations
airlines_ATL_ORD_Q12015 = airlines.filter( (airlines.QUARTER  == 1) & (airlines.YEAR  == 2015) & \
               ((airlines.ORIGIN  == 'ORD') | (airlines.ORIGIN  == 'ATL')))

# COMMAND ----------

def split_city_name(city_state):
  '''UDF to deal with cases where dual cities are labeled
  with a "/". Returns only first city '''

  city = city_state.split(',')[0]
  state = city_state.split(',')[1]
  shortened_city = city.split('/')[0]
  
  return shortened_city + ',' + state

split_city_name_udf = f.udf(split_city_name, types.StringType())

airlines_ATL_ORD_Q12015_short_city = airlines_ATL_ORD_Q12015.withColumn("SHORT_DEST_CITY_NAME", split_city_name_udf('DEST_CITY_NAME'))

# COMMAND ----------

#SAVE FILE TO DBFS
airlines_ATL_ORD_Q12015_short_city.write.format("parquet").save(final_project_path + "airlines_ATL_ORD_Q12015_mod.parquet")

# COMMAND ----------

# MAGIC %md #### READ FROM SAVED FILES ---START FROM HERE---

# COMMAND ----------

weather_q12015_coded = spark.read.parquet(final_project_path+"weather_q12015_coded.parquet")

# COMMAND ----------

airlines_ATL_ORD_Q12015_saved = spark.read.parquet(final_project_path+"airlines_ATL_ORD_Q12015_mod.parquet")

# COMMAND ----------

# MAGIC %md #### Round Airline Scheduled time and Weather data to nearest hour

# COMMAND ----------

#Add city state combined column to timezone reference table

def join_city_state(city, state):
  city_state = str(city) + ', ' + str(state)
  
  return city_state

join_city_state_udf = f.udf(join_city_state, types.StringType())

city_state_timezone = city_timezone.withColumn("city_state", join_city_state_udf('_c1', '_c3'))


# COMMAND ----------

#Join timezone to airline data for origin and destination

airlines_ATL_ORD_Q12015_dest_timezone = airlines_ATL_ORD_Q12015_saved.join(city_state_timezone[['city_state','_c4']], 
                            (airlines_ATL_ORD_Q12015_saved.SHORT_DEST_CITY_NAME == city_state_timezone.city_state),
                            'left')
airlines_ATL_ORD_Q12015_dest_timezone = airlines_ATL_ORD_Q12015_dest_timezone.withColumnRenamed("_c4","DEST_TIMEZONE")
airlines_ATL_ORD_Q12015_dest_timezone = airlines_ATL_ORD_Q12015_dest_timezone.drop('city_state')

# COMMAND ----------

airlines_ATL_ORD_Q12015_timezone = airlines_ATL_ORD_Q12015_dest_timezone.join(city_state_timezone[['city_state','_c4']], 
                             (airlines_ATL_ORD_Q12015_dest_timezone.ORIGIN_CITY_NAME == city_state_timezone.city_state),
                             'left')
airlines_ATL_ORD_Q12015_timezone = airlines_ATL_ORD_Q12015_timezone.withColumnRenamed("_c4","ORIGIN_TIMEZONE")
airlines_ATL_ORD_Q12015_timezone = airlines_ATL_ORD_Q12015_timezone.drop('city_state')

# COMMAND ----------

display(airlines_ATL_ORD_Q12015_timezone)

# COMMAND ----------

# THIS SECTION TAKES CRS (SCHEDULED) DEP TIME AND DEPARTURE CITY AND CREATES TWO COLUMNS- UTC TIME AND TIMEZONE CITY. THESE CAN BE USED TO MERGE WITH WEATHER DATA WHICH ONLY HAS UTC TIME.
#STILL TO RESOLVE- HOW DO WE MATCH UP CITIES TO THEIR TIMEZONE CITY (EG- ATLANTA TO NEW YORK)? MAYBE THERE IS AN EXTERNAL REFERENCE TABLE FOR THIS.

def truncate_time (year, month, day, CRS_DEP_TIME):
  hour = int(str(CRS_DEP_TIME)[:-2])
  return f"{year}-{month}-{day}T{hour}:00:00"

def subtract_time(datetime_original, hours_to_subtract=2):
  datetime_minus_2 = datetime_original - timedelta(hours = hours_to_subtract)
  print(datetime_minus_2)
  return datetime_minus_2

truncate_scheduled_deptime_udf = f.udf(truncate_time, types.StringType())
subtract_time_udf = f.udf(subtract_time, types.StringType())

airlines_ATL_ORD_Q12015_rounded_utc = airlines_ATL_ORD_Q12015_timezone.withColumn("TRUNCATED_CRS_DEP_TIME", truncate_scheduled_deptime_udf('YEAR', 'MONTH', 'DAY_OF_MONTH', 'CRS_DEP_TIME').cast(types.TimestampType()))\
                                      .withColumn("TRUNCATED_CRS_DEP_TIME_UTC",to_utc_timestamp(col("TRUNCATED_CRS_DEP_TIME"), col('ORIGIN_TIMEZONE')))\
                                      
airlines_ATL_ORD_Q12015_full_utc = airlines_ATL_ORD_Q12015_rounded_utc.withColumn("TRUNCATED_CRS_DEP_MINUS_THREE_UTC", airlines_ATL_ORD_Q12015_rounded_utc.TRUNCATED_CRS_DEP_TIME_UTC - f.expr('INTERVAL 3 HOURS'))


# COMMAND ----------

display(airlines_ATL_ORD_Q12015_full_utc)

# COMMAND ----------

#ADD TRUNCATED UTC TIME TO WEATHER DATA DOWN TO HOUR
weather_q12015_coded_rounded = weather_q12015_coded.withColumn("hour", f.date_trunc('hour',f.to_timestamp("DATE","yyyy-MM-ddTHH:mm:ss 'UTC'")))


# COMMAND ----------

display(weather_q12015_coded_rounded)

# COMMAND ----------

# MAGIC %md #### JOIN FLIGHT AND WEATHER DATA

# COMMAND ----------

#JOIN WEATHER AND AIRLINE DATA FOR ORIGIN BETWEEN 2-3 hours DEFORE DEPARTURE
weather_airline_joined_origin = airlines_ATL_ORD_Q12015_full_utc.join(weather_q12015_coded_rounded, (airlines_ATL_ORD_Q12015_full_utc.ORIGIN == weather_q12015_coded_rounded.airport_code) &\
                                         (airlines_ATL_ORD_Q12015_full_utc.TRUNCATED_CRS_DEP_MINUS_THREE_UTC == weather_q12015_coded_rounded.hour), 'left')

# COMMAND ----------

#JOIN WEATHER AND AIRLINE DATA FOR DESTINATION BETWEEN 2-3 hours DEFORE DEPARTURE
weather_airline_joined_full = weather_airline_joined_origin.join(weather_q12015_coded_rounded, (weather_airline_joined_origin.DEST == weather_q12015_coded_rounded.airport_code) &\
                                         (weather_airline_joined_origin.TRUNCATED_CRS_DEP_MINUS_THREE_UTC == weather_q12015_coded_rounded.hour), 'left')

# COMMAND ----------

display(weather_airline_joined)

# COMMAND ----------

#SAVE JOINED FILE
weather_airline_joined.write.format("parquet").save(final_project_path + "weather_airline_joined.parquet")

# COMMAND ----------

#READ BACK JOINED FILE
weather_airline_joined = spark.read.parquet(final_project_path+"weather_airline_joined.parquet")

# COMMAND ----------

display(weather_airline_joined)

# COMMAND ----------

# MAGIC %md
# MAGIC # Next steps
# MAGIC - Join weather data for ALL cities, not just Atlanta, Chicago
# MAGIC - Weather data should be joined at 2 hour lag
# MAGIC - Change time to at least 2 hours before
# MAGIC 
# MAGIC # Features
# MAGIC - How many delays at airport before this flight?
# MAGIC - Incoming flight delay
# MAGIC - Weather at departure station and arrival station
# MAGIC 
# MAGIC 
# MAGIC # Building Model
# MAGIC - Random Forest
# MAGIC - Logistic Regression

# COMMAND ----------

# MAGIC %md ## EDA

# COMMAND ----------

weather_airline_joined.printSchema()

# COMMAND ----------

delay_columns = ['DEP_DELAY','DEP_DEL15']
delay = weather_airline_joined.select(*delay_columns)

# COMMAND ----------

delay.describe().show()

# COMMAND ----------

def plot_hist(labels,values):
    df = pd.DataFrame({'lab':labels, 'val':values})
    df.plot.bar(x='lab', y='val', rot=0)

# COMMAND ----------

# We will use filter instead of rdd, and take advantage of predicate pushdown
import math
def makeHistogram(_min,_max,numBuckets,colName):
    _range = list(range(math.floor(_min), math.ceil(_max), round((abs(_min)+abs(_max))/numBuckets)))
    _counts = np.zeros(len(_range))
    for idx, val in enumerate(_range):
        if idx < len(_range)-1:
            _counts[idx] = delay.filter(F.col(colName) >= _range[idx]) \
                               .filter(F.col(colName) <= _range[idx+1]) \
                               .count()
    plot_hist(_range,_counts)

# COMMAND ----------

# MAGIC %%time
# MAGIC display(makeHistogram(-28.0,1221,11,'DEP_DELAY'))

# COMMAND ----------

delay_histogram = delay.select('DEP_DELAY').rdd.flatMap(lambda x: x).histogram(11)

# COMMAND ----------

# Loading the Computed Histogram into a Pandas Dataframe for plotting
pd.DataFrame(
    list(zip(*delay_histogram)), 
    columns=['bin', 'frequency']
).set_index(
    'bin'
).plot(kind='bar');


# COMMAND ----------

# MAGIC %md # Stations

# COMMAND ----------

stations = spark.read.option("header", "true").csv("dbfs:/mnt/mids-w261/DEMO8/gsod/stations.csv.gz")

# COMMAND ----------

display(stations.describe())

# COMMAND ----------

from pyspark.sql import functions as f
stations.where(f.col('name').contains('ATLANTA HARTSFIELD INTERNATIONAL AIRPORT, GA US'))

# COMMAND ----------

display(stations.where(f.col('call').contains('LAX')))


# COMMAND ----------

stations.select('name').distinct().count()

# COMMAND ----------

display(stations.select('name').distinct())

# COMMAND ----------

weather.select('NAME').distinct().count()

# COMMAND ----------

display(weather.select('name').distinct())

# COMMAND ----------

