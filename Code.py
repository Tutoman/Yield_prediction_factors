# Databricks notebook source
# MAGIC %md
# MAGIC ## Big Data Tools Group Assignment - YELP Analysis
# MAGIC 
# MAGIC Big Data Tools 2 Course IESEG 2021/2022 Group 13
# MAGIC 
# MAGIC - ANGGORO Fajar Tri
# MAGIC - MANTILLA Omar
# MAGIC - VELLINGIRIKOWSALYA Swasthik
# MAGIC 
# MAGIC 
# MAGIC The goal of this project is to identify what are some of the most important factors that causes business to implement "delivery or takeaway" 
# MAGIC services during the pandemic. We do this by Analysing the Data & Developing a predictive Machine Learning Model. The structure of this project is as follows:
# MAGIC 
# MAGIC - Data Cleaning and Basetable creation
# MAGIC - Exploratory Data Analysis
# MAGIC - Model Development
# MAGIC - Model Evaluation and selection
# MAGIC - Conclusion

# COMMAND ----------

from pyspark.sql.functions import *

# Set up Path
path_business = "/FileStore/tables/parsed_data/parsed_business.json"
path_checkin = "/FileStore/tables/parsed_data/parsed_checkin.json"
path_covid = "/FileStore/tables/parsed_data/parsed_covid.json"
path_review = "/FileStore/tables/parsed_data/parsed_review.json"
path_tip = "/FileStore/tables/parsed_data/parsed_tip.json"
path_user = "/FileStore/tables/parsed_data/parsed_user.json"

# this setting is necessary for timeparser
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY") 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Cleaning Process
# MAGIC We will read all the data and clean each table separately

# COMMAND ----------

# Read Business Data
business = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(path_business)

business.show(2)

# COMMAND ----------

# check schema
business.printSchema()

# COMMAND ----------

# since . are a special case and more difficult to process, we will replace dots in our columns
new_cols=(column.replace('.', '_') for column in business.columns)
bs2 = business.toDF(*new_cols)

# select columns that contains certain string
selected = [s for s in bs2.columns if 'hours' in s]+['business_id', 'name']
bs2.select(selected).show(5)

# COMMAND ----------

# what are the different categories available?
bs2.select("categories").distinct().show(50,truncate = False)


# COMMAND ----------

# DBTITLE 1,NUMBER OF BUSINESS PER CATEGORIES
business_cate = bs2.groupBy("categories").agg(count(col("business_id")).alias("Total_business"))
business_cate.orderBy(col("Total_business").desc()).display()

# COMMAND ----------

# lets try to filter business categories
bs2 = bs2.filter( (col("categories").contains("Restaurants")) | (col("categories").contains("Food")) | (col("categories").contains("Bars")) )

# how many business do we have after the filter?
print((bs2.count(), len(bs2.columns)))

# COMMAND ----------

# we have a lot of columns, how do we pick them?
# 1st step is to always drop columns with 0 variance (same values in a column)


# use countdistinct function        
expr = [countDistinct(c).alias(c) for c in bs2.columns]

# select columns 
countdf =  bs2.select(*expr)

# if the value is less than 1, meaning there are only nulls, drop it
cols2drop = [k for k,v in countdf.collect()[0].asDict().items() if v < 1]

# what is the total column number after filtering 0 variances?
len(bs2.drop(*cols2drop).columns)

# COMMAND ----------

# what is the column that only contains 1 value?
for cols in bs2.columns:
     if cols not in bs2.drop(*cols2drop).columns:
            print(cols)
     else:
        continue

# COMMAND ----------

# to preprocess the business table, we just need to drop columns and fill NAs (no aggregation needed)

# from our previous 0 variances evaluation
bs2 = bs2.drop(*cols2drop)

# manual evaluation on which columns to keep
cols_keep = ['attributes_AcceptsInsurance', 'attributes_AgesAllowed', 'attributes_Alcohol', 'attributes_Ambience', 'attributes_BYOB', 
             'attributes_BYOBCorkage', 'attributes_BestNights', 'attributes_BikeParking', 'attributes_BusinessAcceptsBitcoin', 
             'attributes_BusinessAcceptsCreditCards', 'attributes_BusinessParking', 'attributes_ByAppointmentOnly', 'attributes_Caters', 
             'attributes_CoatCheck', 'attributes_Corkage', 'attributes_DietaryRestrictions', 'attributes_DogsAllowed', 
             'attributes_DriveThru', 'attributes_GoodForDancing', 'attributes_GoodForKids', 'attributes_GoodForMeal', 
             'attributes_HappyHour', 'attributes_HasTV', 'attributes_Music', 'attributes_NoiseLevel', 
             'attributes_Open24Hours', 'attributes_OutdoorSeating', 'attributes_RestaurantsAttire', 'attributes_RestaurantsCounterService', 
             'attributes_RestaurantsDelivery', 'attributes_RestaurantsGoodForGroups', 'attributes_RestaurantsPriceRange2', 
             'attributes_RestaurantsReservations', 'attributes_RestaurantsTableService', 'attributes_RestaurantsTakeOut', 
             'attributes_Smoking', 'attributes_WheelchairAccessible', 'attributes_WiFi', 'business_id', 'city', 'is_open', 
             'review_count', 'stars', 'state']

# COMMAND ----------

# only keep the columns to keep
bs2 = bs2.select(cols_keep)
len(bs2.columns)

# COMMAND ----------

# we will process the general column first, that is, columns that doesnt require specific processing
gen_cols = ['attributes_AcceptsInsurance',  'attributes_BYOB', 
             'attributes_BikeParking', 'attributes_BusinessAcceptsBitcoin', 
             'attributes_BusinessAcceptsCreditCards',  'attributes_ByAppointmentOnly', 'attributes_Caters', 
             'attributes_CoatCheck', 'attributes_Corkage', 'attributes_DogsAllowed', 
             'attributes_DriveThru', 'attributes_GoodForDancing', 'attributes_GoodForKids',  
             'attributes_HappyHour', 'attributes_HasTV',  'attributes_Open24Hours', 'attributes_OutdoorSeating', 
             'attributes_RestaurantsCounterService', 
             'attributes_RestaurantsDelivery', 'attributes_RestaurantsGoodForGroups',  
             'attributes_RestaurantsReservations', 'attributes_RestaurantsTableService', 'attributes_RestaurantsTakeOut', 
              'attributes_WheelchairAccessible', 'attributes_RestaurantsPriceRange2','business_id']

# do this with pandas
import pandas as pd

temp = bs2.select(gen_cols).toPandas()

# label encoding
for i in temp.columns[:-2]:
    temp[i] = temp[i].apply(lambda x: 1 if x == 'True' else 0)
    
    # check if certain column only contains 1 unique
    if temp[i].nunique() == 1:
        print(i)

# handle missing values for the price column: insert middle value, however we cant do this directly since this column is a string
temp['attributes_RestaurantsPriceRange2'].fillna('None', inplace = True)

# insert middle value which is 2.5
temp['attributes_RestaurantsPriceRange2'] = temp['attributes_RestaurantsPriceRange2'].apply(lambda x: '2.5' if x == 'None' else x)

# change datatype into float
temp['attributes_RestaurantsPriceRange2'] = temp['attributes_RestaurantsPriceRange2'].astype('float')

temp.head()

# COMMAND ----------

# move business id into the first column

first_column = temp.pop('business_id')
  
# insert column using insert(position,column_name, first_column) function
temp.insert(0, 'business_id', first_column)

temp.head()

# COMMAND ----------

# these are the columns that require 'special' processing
# lets fix these columns
bs2.select("attributes_Alcohol", "attributes_Ambience", 'attributes_BestNights', 'attributes_Music','attributes_BusinessParking', 'attributes_GoodForMeal', "attributes_AgesAllowed", "attributes_DietaryRestrictions", "attributes_Smoking", "attributes_WiFi").show(10)


# COMMAND ----------

# lets fix the alcohol column, we will create a flag variable whether a business serves alcohol or not

import pandas as pd

alcohol = bs2.select("business_id", "attributes_Alcohol").dropna().toPandas()

alcohol['doesnt_serve_alcohol'] = alcohol["attributes_Alcohol"].str.contains("none" ) 
alcohol['doesnt_serve_alcohol'] = alcohol['doesnt_serve_alcohol'].astype('int')

alcohol.drop("attributes_Alcohol", axis = 1, inplace = True)

alcohol.head()

# COMMAND ----------

# fix ambience column by evaluating if a business has specific ambience corresponds to it

amb = bs2.select("business_id", "attributes_Ambience").dropna().toPandas()

amb['has_ambience'] = amb["attributes_Ambience"].str.contains("True" ) 
amb['has_ambience'] = amb['has_ambience'].astype('int')

amb.drop("attributes_Ambience", axis = 1, inplace = True)

amb.head()

# COMMAND ----------

# fix best nights column by evaluating if a business has specific best nights corresponds to it

bn = bs2.select("business_id", 'attributes_BestNights').dropna().toPandas()

bn['has_specific_bestnight'] = bn['attributes_BestNights'].str.contains("True" ) 
bn['has_specific_bestnight'] = bn['has_specific_bestnight'].astype('int')

bn.drop("attributes_BestNights", axis = 1, inplace = True)

bn.head()

# COMMAND ----------

# fix music column by evaluating if a business has a Music

ms = bs2.select("business_id", 'attributes_Music').dropna().toPandas()

ms['has_music'] = ms['attributes_Music'].str.contains("True" ) 
ms['has_music'] = ms['has_music'].astype('int')

ms.drop('attributes_Music', axis = 1, inplace = True)

ms.head()

# COMMAND ----------

# fix business parking column by evaluating if a business has a parking

park = bs2.select("business_id", 'attributes_BusinessParking').dropna().toPandas()

park['has_parking'] = park['attributes_BusinessParking'].str.contains("True" ) 
park['has_parking'] = park['has_parking'].astype('int')

park.drop('attributes_BusinessParking', axis = 1, inplace = True)

park.head()

# COMMAND ----------

# fix goodformeal column by evaluating if a business has specific Meal speciality corresponds to it

meal = bs2.select("business_id", 'attributes_GoodForMeal').dropna().toPandas()

meal['has_specific_goodmeal'] = meal['attributes_GoodForMeal'].str.contains("True" ) 
meal['has_specific_goodmeal'] = meal['has_specific_goodmeal'].astype('int')

meal.drop('attributes_GoodForMeal', axis = 1, inplace = True)

meal.head()

# COMMAND ----------

# fix agescolumn  by evaluating if a business has age restrictions corresponds to it

age = bs2.select("business_id", "attributes_AgesAllowed").dropna().toPandas()

age['has_age_restrict'] = age['attributes_AgesAllowed'].str.contains("plus" ) 
age['has_age_restrict'] = age['has_age_restrict'].astype('int')

age.drop("attributes_AgesAllowed", axis = 1, inplace = True)

age.head()

# COMMAND ----------

# fix diet column by evaluating if a business has diet option available to it

diet = bs2.select("business_id", "attributes_DietaryRestrictions").dropna().toPandas()

diet['has_diet_option'] = diet["attributes_DietaryRestrictions"].str.contains("True" ) 
diet['has_diet_option'] = diet['has_diet_option'].astype('int')

diet.drop("attributes_DietaryRestrictions", axis = 1, inplace = True)

diet.head()

# COMMAND ----------

# fix smoking column by evaluating if a business allows smoking 

smoke = bs2.select("business_id", "attributes_Smoking").dropna().toPandas()

smoke['allows_smoking'] = smoke["attributes_Smoking"].str.contains("yes|outdoor" ) 
smoke['allows_smoking'] = smoke['allows_smoking'].astype('int')

smoke.drop("attributes_Smoking", axis = 1, inplace = True)

smoke.head()

# COMMAND ----------

# fix wifi column by evaluating if a business has a wifi 

wifi = bs2.select("business_id", "attributes_WiFi").dropna().toPandas()

wifi['has_wifi'] = wifi["attributes_WiFi"].str.contains("paid|free" ) 
wifi['has_wifi'] = wifi['has_wifi'].astype('int')

wifi.drop("attributes_WiFi", axis = 1, inplace = True)

wifi.head()

# COMMAND ----------

# The next step is to combine these special columns, into our original business dataframe

temp = temp.merge(alcohol, how = 'left', on = 'business_id')
temp = temp.merge(amb, how = 'left', on = 'business_id')
temp = temp.merge(bn, how = 'left', on = 'business_id')
temp = temp.merge(ms, how = 'left', on = 'business_id')
temp = temp.merge(park, how = 'left', on = 'business_id')
temp = temp.merge(meal, how = 'left', on = 'business_id')
temp = temp.merge(age, how = 'left', on = 'business_id')
temp = temp.merge(diet, how = 'left', on = 'business_id')
temp = temp.merge(smoke, how = 'left', on = 'business_id')
temp = temp.merge(wifi, how = 'left', on = 'business_id')

# for the columns that had just been merged, fill Nas with 0
temp.fillna(0, inplace = True)

temp.head()

# COMMAND ----------

# convert the dataframe back into spark DF

df1 = spark.createDataFrame(temp)
df1.printSchema()
df1.show(5)

# COMMAND ----------

# this is the second part of the data, we will merge this with the processed previous business DF
df2 = bs2.select('business_id', 'city', 'is_open', 'review_count', 'stars', 'state')

df2.show(5)

# COMMAND ----------

# create business table (processed) by joining df1 & df2
business_proc = df2.join(df1, on = "business_id", how = "left")

business_proc.show(2)

# COMMAND ----------

# granularity checking
business_proc.select(countDistinct("business_id")).show()

print((business_proc.count(), len(business_proc.columns)))

# COMMAND ----------

# Read checkin Data
checkin = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(path_checkin)

checkin.show(5)

# COMMAND ----------

#checkin.show(5, truncate = False)
from pyspark.sql.functions import countDistinct

# How many Business have been checked in at least once?
checkin.select(countDistinct("business_id")).show()

# COMMAND ----------

# lets process the checkin table by counting how many checkins each business have
checkin_proc = checkin.groupBy("business_id").agg(count("date").alias("count_checkins"), last("date").alias("last_checkin"))

# calculate checkin recency
checkin_proc = checkin_proc.withColumn("checkin_recency_months", round(months_between(to_date(lit("2020-04-01"), "yyyy-MM-dd"), to_date(col("last_checkin"), "yyyy-MM-dd HH:mm:ss")) ,0))

checkin_proc = checkin_proc.drop("last_checkin")
                                       
checkin_proc.show(10)

# COMMAND ----------

# Read covid Data
covid = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(path_covid)

covid.show(5)

# COMMAND ----------

# How many businesess?
covid.select(countDistinct("business_id")).show()

# process the covid dataset by taking the necessary column
covid_proc = covid.select("business_id", "Temporary Closed Until", "delivery or takeout")

# since " " is more difficult to process, we will replace them in our columns
new_cols=(column.replace(' ', '_') for column in covid_proc.columns)
covid_proc = covid_proc.toDF(*new_cols)

# Removing the business that are "closed until" as they dont help for the predictions while they are closed
covid_proc = covid_proc.filter(covid_proc.Temporary_Closed_Until == "FALSE")
covid_proc = covid_proc.drop("Temporary_Closed_Until")

# add label, essentially the 1 0 version of delivery take out column
covid_proc = covid_proc.withColumn("label", when(covid_proc.delivery_or_takeout == 'TRUE', 1).otherwise(0) )

# drop duplicates
covid_proc = covid_proc.dropDuplicates(subset = ["business_id"])

covid_proc.show(5)

# COMMAND ----------

# Read user Data
user = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(path_user)

user.show(5)

# COMMAND ----------

# DBTITLE 1,NUMBER OF USERS PER YEARS YELPING
from pyspark.sql.functions import datediff,col
from pyspark.sql.functions import *
user1=user.withColumn("date2", lit("2020-4-1"))
user1=user1.withColumn("date2",to_date("date2","yyyy-MM-dd"))\
           .withColumn("yelping_since",to_date("yelping_since","yyyy-MM-dd"))\
           .withColumn("diff_in_years", datediff(col("date2"),col("yelping_since"))/365.25)\
           .withColumn("datesDiff", datediff(col("date2"),col("yelping_since"))) \
           .withColumn("montsDiff", months_between(col("date2"),col("yelping_since"))) \
           .withColumn("montsDiff_round",round(months_between(col("date2"),col("yelping_since")),2)) \
           .withColumn("yearsDiff",months_between(col("date2"),col("yelping_since"))/lit(12)) \
           .withColumn("Years",round(months_between(col("date2"),col("yelping_since"))/lit(12),0))

user_yelping_years = user1.groupBy("Years").agg(count(col("user_id")).alias("Total_users"))
user_yelping_years.orderBy(col("Total_users").desc()).display()

# COMMAND ----------

# first, lets only take variables that we think are going to be useful
user = user.select("user_id", col("average_stars").alias("average_reviewer_stars") , "fans", "review_count", to_date(col("yelping_since"), "yyyy-MM-dd HH:mm:ss").alias("yelping_since"),\
                  round(months_between(to_date(lit("2020-04-01"), "yyyy-MM-dd"), col("yelping_since")),0).alias("user_LOR_months"))

user = user.drop("yelping_since")

user.show(5)

# COMMAND ----------

print((user.count(), len(user.columns)))

# COMMAND ----------

# Read review Data
review = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(path_review)

review.show(5)

# COMMAND ----------

# How many businesess are reviewed?
review.select(countDistinct("business_id")).show()

# How many reviews are there?
review.select(countDistinct("review_id")).show()

# How many users giving reviews?
review.select(countDistinct("user_id")).show()

# COMMAND ----------

# describe dataset
review.describe().select("summary", "date", "cool", "funny", "stars", "useful").show()

# COMMAND ----------

# before we process the review data, we need to merge user data into this table, this is because reviews table is the only table to connect business table and user table

review_proc = review.select("business_id", "date", "stars", "useful", "user_id")

review_proc = review_proc.join(user, on = "user_id", how = "left")

review_proc.show(5)

# COMMAND ----------

# lets process review data by gathering average cool, funny, usefullness, and stars each business have in terms of reviews
review_proc = review_proc.groupBy("business_id")\
    .agg(avg("stars").alias("avg_business_stars"), \
         avg("useful").alias("avg_useful_reviews_count"), \
         count("useful").alias("count_reviews"), \
         countDistinct("user_id").alias("count_reviewers"), \
         last("date").alias("last_review"), \
         avg("fans").alias("avg_reviewer_fanscount"), \
         avg("review_count").alias("avg_reviewers_reviewcount"), \
         avg("user_LOR_months").alias("avg_reviewers_LOR") \
     )

review_proc = review_proc.withColumn("review_recency_months", round(months_between(to_date(lit("2020-04-01"), "yyyy-MM-dd"), to_date(col("last_review"), "yyyy-MM-dd HH:mm:ss")) ,0))

review_proc = review_proc.drop("last_review")

review_proc.show(5) 

# COMMAND ----------

# Read tip Data
tip = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(path_tip)

tip.show(5)

# COMMAND ----------

# describe dataset
tip.describe().select("summary", "date", "compliment_count").show()

# COMMAND ----------

# How many businesess are tipped?
tip.select(countDistinct("business_id")).show()

tip.select("compliment_count").distinct().show()

# COMMAND ----------

# lets process tip data by the average compliments count each business have in terms of tips
tip_proc = tip.select("business_id", "date").groupBy("business_id")\
    .agg(last("date").alias("last_tip"), \
         count("date").alias("count_tip") \
     )

tip_proc = tip_proc.withColumn("tip_recency_months", round(months_between(to_date(lit("2020-04-01"), "yyyy-MM-dd"), to_date(col("last_tip"), "yyyy-MM-dd HH:mm:ss")) ,0))

tip_proc = tip_proc.drop("last_tip")

tip_proc.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Merging Process & Basetable creation
# MAGIC After done processing all the tables, we will now merge all of them and form a basetable

# COMMAND ----------

# Our starting table is the business table

# business with checkin
basetable = business_proc.join(checkin_proc, on = "business_id", how = "left")

# with reviews
basetable = basetable.join(review_proc, on = "business_id", how = "left")

# with tip
basetable = basetable.join(tip_proc, on = "business_id", how = "left")

# with covid
basetable = basetable.join(covid_proc, on = "business_id", how = "left")

# COMMAND ----------

basetable.show(2)

# COMMAND ----------

# evaluate datatypes
basetable.printSchema()

# COMMAND ----------

# shape
print((basetable.count(), len(basetable.columns)))

# granularity checking
basetable.select(countDistinct("business_id")).show()

# COMMAND ----------

# check if there are null values
basetable.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in basetable.columns]).show()

# COMMAND ----------

basetable.select("business_id", "count_reviews", "review_count", "avg_business_stars", "stars").show()

# COMMAND ----------

# we will handle the missing values

# 0 for count variables
basetable = basetable.na.fill(0, subset=["count_checkins","count_tip"])

# max value for recency variables
max_recency_checkin = basetable.select(max("checkin_recency_months")).collect()[0][0]
basetable = basetable.na.fill(max_recency_checkin, subset=["checkin_recency_months"])

max_recency_tip = basetable.select(max("tip_recency_months")).collect()[0][0]
basetable = basetable.na.fill(max_recency_tip, subset=["tip_recency_months"])

# COMMAND ----------

# drop NA for businesses that is closed during covid
basetable = basetable.na.drop()

# COMMAND ----------

# export the basetable into json format
basetable.write.format("json")\
  .mode("overwrite")\
  .save("/FileStore/tables/parsed_data/basetable.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC In this stage, we're going to explore deeper into the data and try to understand are there any relation between our variables and the business starting doing delivery or takeouts

# COMMAND ----------

# path
path_base = "/FileStore/tables/parsed_data/basetable.json"

# Read basetable
basetable = spark.read.format("json")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(path_base)

#basetable.show(5)

# COMMAND ----------

# DBTITLE 1,NUMBER OF BUSINESS PER STAR RATE
# Business per Starts
business_st = bs2.groupBy("stars").agg(count(col("business_id")).alias("Total Business"))
business_st.orderBy(col("stars").desc()).display()

# COMMAND ----------

# DBTITLE 1,BUSINESS PER STATE
# Business per state
business_state = bs2.groupBy("state").agg(count(col("business_id")).alias("Total Business"))
business_state.orderBy(col("Total Business").desc()).display()

# COMMAND ----------

# DBTITLE 1,BUSINESS PER CITY
# Business per city
business_city = bs2.groupBy("city").agg(count(col("business_id")).alias("Total_business"))
business_city.orderBy(col("Total_business").desc()).display()

# COMMAND ----------

# Reviews
# Number of reviews based on the stars
review.groupby("stars").agg(count("business_id").alias("Number of Reviews")).sort("stars").display()

# COMMAND ----------

# Top 10 businesses with highest number of reviews
review.groupby("business_id").agg(count("business_id").alias("Number of Reviews")).sort(desc("Number of Reviews")).limit(10).display()

# COMMAND ----------

# Tips
# Total Number of tips based on the compliments received. 
tip.groupby("compliment_count").count().sort("compliment_count").display()

# COMMAND ----------

# Top 10 businesses based on the number of tips.
tip.groupby("business_id").agg(count("business_id").alias("Number of Tips")).sort(desc("Number of Tips")).limit(10).display()

# COMMAND ----------

## Analysis made for Business Section
# Analysis Restaurants Takeout before covid and after covid
basetable.groupBy("attributes_RestaurantsTakeOut") \
  .agg(count("attributes_RestaurantsTakeOut").alias("Before Covid"),sum("label").alias("After Covid")) \
  .withColumnRenamed("attributes_RestaurantsTakeOut", "TakeOut Availability") \
  .display()

# COMMAND ----------

  ## Takeaway/Delivery based on kid friendly restaurants
basetable.groupBy("attributes_GoodForKids") \
  .agg(count("attributes_GoodForKids").alias("Total_Businesses"),sum("label").alias("Takeaway/Delivery After COVID")) \
  .withColumn('attributes_GoodForKids', regexp_replace('attributes_GoodForKids', '0', 'Not Good')) \
  .withColumn('attributes_GoodForKids', regexp_replace('attributes_GoodForKids', '1', 'Good')) \
  .withColumnRenamed("attributes_GoodForKids", "Good For Kids") \
  .display()

# COMMAND ----------

## Total Number of Deliveries/Takeaways for businesses that previously did not do delivery
basetable.filter(basetable.attributes_RestaurantsDelivery == 0).groupBy("label") \
  .count().alias("After Covid") \
  .display()

# COMMAND ----------

##Average number of total reviews based on a business that does delivery/takeaway or not
basetable.filter(basetable.attributes_RestaurantsDelivery == 0).groupBy("label") \
  .avg("review_count").alias("Average Reviews") \
  .display()

# COMMAND ----------

## % of restaurants that does delivery/takeaway based on the price range
basetable.groupBy("attributes_RestaurantsPriceRange2") \
  .agg(count("label").alias("Total_Businesses"),sum("label").alias("Delivery/Takeaway")) \
  .withColumn("% Availability",round((col("Delivery/Takeaway")/col("Total_Businesses"))*100,2)) \
  .withColumnRenamed("attributes_RestaurantsPriceRange2", "Price Range") \
  .orderBy("Price Range") \
  .display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Development
# MAGIC In this step, we will develop different Machine learning Models

# COMMAND ----------

# drop some columns
basetable = basetable.drop("delivery_or_takeout") # original covid column
basetable = basetable.drop("count_reviews", "avg_business_stars") # drop calculated rating & review, we'll use the original data

# COMMAND ----------

# Now let's process our data for ML

# process state & city, use string indexer and One hot encoding
from pyspark.ml.feature import OneHotEncoder, StringIndexer

# we will also need to scale our data
from pyspark.ml.feature import MinMaxScaler

# assembler will help us put all features into one column
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline

#state
stateIndxr = StringIndexer().setInputCol("state").setOutputCol("stateInd")

#city
cityIndxr = StringIndexer().setInputCol("city").setOutputCol("cityInd")

#One-hot encoding
ohee_catv = OneHotEncoder(inputCols=["stateInd","cityInd"],outputCols=["state_ohe","city_ohe"])

pipe_catv = Pipeline(stages=[stateIndxr, cityIndxr, ohee_catv])

basetable = pipe_catv.fit(basetable).transform(basetable)

basetable = basetable.drop("stateInd","cityInd")

basetable.select("state_ohe", "city_ohe").show(3)

# COMMAND ----------

#Drop state and city from the data
basetable = basetable.drop("city", "state")

basetable.printSchema()

# COMMAND ----------

# lets split our Data into Training & Testing

# first, check the distribution
basetable.groupBy("label").count().show()

# since we have uneven distribution of class, we will do a stratified sampling
# pyspark doesnt have this convenient method, therefore we will do it manually
# source: https://stackoverflow.com/questions/47637760/stratified-sampling-with-pyspark

# Taking 70% of both 0's and 1's into training set
train = basetable.sampleBy("label", fractions={0: 0.7, 1: 0.7}, seed=42)

# Subtracting 'train' from original 'data' to get test set 
test = basetable.subtract(train)

# recheck the distribution
train.groupBy("label").count().show()
test.groupBy("label").count().show()

# COMMAND ----------

#Transform the tables in a table of label, features format
from pyspark.ml.feature import RFormula

train = RFormula(formula="label ~ . - business_id").fit(train).transform(train)
test = RFormula(formula="label ~ . - business_id").fit(test).transform(test)

print("train nobs: " + str(train.count()))
print("test nobs: " + str(test.count()))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1st Model : Logistic Regression
# MAGIC A Logistic Regression Model build through Grid Search, 10 fold CV with AUC as deciding criteria

# COMMAND ----------

## Modelling : Logistic Regression

from pyspark.ml.classification import LogisticRegression

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Define pipeline
lr = LogisticRegression()
pipeline = Pipeline().setStages([lr])

#Set param grid
params = ParamGridBuilder()\
  .addGrid(lr.regParam, [0.1, 0.01])\
  .addGrid(lr.maxIter, [50, 100,150])\
  .build()

#Evaluator: uses max(AUC) by default to get the final model
evaluator = BinaryClassificationEvaluator()

#Cross-validation of entire pipeline
lrcv = CrossValidator()\
  .setEstimator(pipeline)\
  .setEstimatorParamMaps(params)\
  .setEvaluator(evaluator)\
  .setNumFolds(10) 

# fitting process
lrModel = lrcv.fit(train)

# COMMAND ----------

#Get predictions on the test set
preds = lrModel.transform(test)
preds.select("prediction", "label", "rawPrediction").show(5)

#Get predictions on the train set
preds_train = lrModel.transform(train)
preds_train.select("prediction", "label", "rawPrediction").show(5)

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics

# define evaluator
evaluator = BinaryClassificationEvaluator()

#Get model accuracy
print("Train accuracy: " + str(evaluator.evaluate(preds_train)))

#Get AUC
metrics_train = BinaryClassificationMetrics(preds_train.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1]))))
print("Train AUC: " + str(metrics_train.areaUnderROC))

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics

# define evaluator
evaluator = BinaryClassificationEvaluator()

#Get model accuracy
print("Test accuracy: " + str(evaluator.evaluate(preds)))

#Get AUC
metrics = BinaryClassificationMetrics(preds.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1]))))
print("Test AUC: " + str(metrics.areaUnderROC))

# COMMAND ----------

#Get more metrics
from pyspark.mllib.evaluation import MulticlassMetrics

labels = preds.select("prediction", "label").rdd.map(lambda lp: lp.label).distinct().collect()

metrics = MulticlassMetrics(preds.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1]))))

for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    #print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
    

# COMMAND ----------

# About our best model
#Get best tuned parameters of pipeline
lrBestPipeline = lrModel.bestModel
BestLRModel = lrBestPipeline.stages[-1]._java_obj.parent() #the stages function refers to the stage in the pipelinemodel

print("Best LR model:")
print("** regParam: " + str(BestLRModel.getRegParam()))
print("** maxIter: " + str(BestLRModel.getMaxIter()))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2nd Model : Decision Tree
# MAGIC A Decision Tree Model build through Grid Search, 10 fold CV with AUC as deciding criteria

# COMMAND ----------

# Modelling: Decision Tree
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# Decision Tree Model
model_dt = DecisionTreeClassifier(labelCol="label",
                                  featuresCol="features", 
                                  maxDepth=2)

# Now lets do a parameter grid for cross validation
paragrid = (ParamGridBuilder().addGrid(model_dt.maxDepth, [2, 5, 10, 20, 30]).addGrid(model_dt.maxBins, [10, 20, 40, 80, 100]).build())

# Lets evaluate the model
evaluator_dt = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

# Lets create 5-fold CrossValidator
dt_cv = CrossValidator(estimator = model_dt, estimatorParamMaps = paragrid, evaluator = evaluator_dt, numFolds = 10)

# Lets get the model with the cv values
Modeldtcv = dt_cv.fit(train)

# COMMAND ----------

#Get predictions on the test set
preds = Modeldtcv.transform(test)
preds.select("prediction", "label", "rawPrediction").show(5)

#Get predictions on the train set
preds_train = Modeldtcv.transform(train)
preds_train.select("prediction", "label", "rawPrediction").show(5)

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics

# define evaluator
evaluator = BinaryClassificationEvaluator()

#Get model accuracy
print("Train accuracy: " + str(evaluator.evaluate(preds_train)))

#Get AUC
metrics_train = BinaryClassificationMetrics(preds_train.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1]))))
print("Train AUC: " + str(metrics_train.areaUnderROC))

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics

# define evaluator
evaluator = BinaryClassificationEvaluator()

#Get model accuracy
print("Test accuracy: " + str(evaluator.evaluate(preds)))

#Get AUC
metrics = BinaryClassificationMetrics(preds.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1]))))
print("Test AUC: " + str(metrics.areaUnderROC))

# COMMAND ----------

#Get more metrics
from pyspark.mllib.evaluation import MulticlassMetrics

labels = preds.select("prediction", "label").rdd.map(lambda lp: lp.label).distinct().collect()

metrics = MulticlassMetrics(preds.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1]))))

for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    #print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
   

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3rd Model : Random Forest
# MAGIC A Random Forest Model build through Grid Search, 10 fold CV with AUC as deciding criteria

# COMMAND ----------

# Modelling: Random Forest
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Define pipeline
rfc = RandomForestClassifier()
rfPipe = Pipeline().setStages([rfc])

#Set param grid
rfParams = ParamGridBuilder()\
  .addGrid(rfc.numTrees, [100, 200, 300])\
  .addGrid(rfc.impurity, ["gini", "entropy"])\
  .addGrid(rfc.maxDepth, [4, 8, 12])\
  .build()

rfCv = CrossValidator()\
  .setEstimator(rfPipe)\
  .setEstimatorParamMaps(rfParams)\
  .setEvaluator(BinaryClassificationEvaluator())\
  .setNumFolds(10) # 10-fold cross validation

#Run cross-validation, and choose the best set of parameters.
rfcModel = rfCv.fit(train)

# COMMAND ----------

#Get predictions on the test set
preds = rfcModel.transform(test)
preds.select("prediction", "label", "rawPrediction").show(5)

# COMMAND ----------

#Get predictions on the train set
preds_train = rfcModel.transform(train)
preds_train.select("prediction", "label", "rawPrediction").show(5)

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics

# define evaluator
evaluator = BinaryClassificationEvaluator()

#Get model accuracy
print("Train accuracy: " + str(evaluator.evaluate(preds_train)))

#Get AUC
metrics_train = BinaryClassificationMetrics(preds_train.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1]))))
print("Train AUC: " + str(metrics_train.areaUnderROC))

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics

# define evaluator
evaluator = BinaryClassificationEvaluator()

#Get model accuracy
print("Test accuracy: " + str(evaluator.evaluate(preds)))

#Get AUC
metrics = BinaryClassificationMetrics(preds.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1]))))
print("Test AUC: " + str(metrics.areaUnderROC))

# COMMAND ----------

#Get more metrics
from pyspark.mllib.evaluation import MulticlassMetrics

labels = preds.select("prediction", "label").rdd.map(lambda lp: lp.label).distinct().collect()

metrics = MulticlassMetrics(preds.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1]))))

for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    #print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
    

# COMMAND ----------

# about our best model

#Select the best RF model
rfcBestModel = rfcModel.bestModel.stages[-1] #-1 means "get last stage in the pipeline"


print("Best RF model:")
#Get tuned number of trees
print("** NumTrees: " + str(rfcBestModel.getNumTrees))

#Get tuned impurity
print("** Impurity: " + str(rfcBestModel.getImpurity()))

#Get tuned maximum depth
print("** MaxDepth: " + str(rfcBestModel.getMaxDepth()))


# COMMAND ----------

#Get feature importances
rfcBestModel.featureImportances

#Prettify feature importances
import pandas as pd
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
  
ExtractFeatureImp(rfcBestModel.featureImportances, train, "features").head(10)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

imp = ExtractFeatureImp(rfcBestModel.featureImportances, train, "features").head(5)
imp = np.round(imp, decimals=2)


fig, ax = plt.subplots(figsize = (4,6))
#sns.set_color_codes('pastel')
ax = sns.barplot(x = 'score', y = 'name', data = imp)
#sns.set_color_codes('muted')
ax.bar_label(ax.containers[0])
ax.set_title('Top 5 Most Important Factors lead Business to do Deliveries or Takeout during Covid')
ax.set_xlabel('Importance Score')
ax.set_ylabel('')
sns.despine(left = True, bottom = True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC Our best model in terms of evaluation metric is **logistic regression**. While some of the most important factors to determine whether a business will start doing deliveries or takeout are: 
# MAGIC - whether that business does deliveries or takeout before 
# MAGIC - whether that business is Good For Kids
# MAGIC - the number of reviews that business has
# MAGIC - The price range
