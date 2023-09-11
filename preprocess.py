from pyspark.sql.functions import col, unix_timestamp
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col
import sys
import os
from pyspark.sql.functions import lit
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import count
import matplotlib.pyplot as plt
from pyspark.sql import Window
from pyspark.sql.functions import col, expr, percentile_approx
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import col, collect_list, expr
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
from tqdm import tqdm

def GetDataset(spark, version='small', split='train', data_dir = ''):
    if version == 'small':
        prefix = '_small'
    else:
        prefix = ''

    interactions = spark.read.parquet(os.path.join(data_dir, f'interactions_{split}{prefix}.parquet'))
    tracks = spark.read.parquet(os.path.join(data_dir, f'tracks_{split}{prefix}.parquet'))
    users = spark.read.parquet(os.path.join(data_dir, f'users_{split}{prefix}.parquet'))

    return interactions, tracks, users

def preprocess(df_tracks, df_interactions):
    # Join the dataframes and select required columns
    df_tracks = df_tracks.select(['recording_msid', 'recording_mbid'])
    initial_joined_df = df_interactions.join(df_tracks, on='recording_msid').select(
        'user_id',
        F.unix_timestamp('timestamp').cast(DoubleType()).alias('timestamp'),
        F.when(F.col("recording_mbid").isNull(), F.col("recording_msid"))
        .otherwise(F.col("recording_mbid")).alias("recording_id")
    )
  
    return initial_joined_df


def split_on_timestamp(final_joined_df, threshold = 0.6, drop_low_interactions = False):
    # Calculate the 60th percentile timestamp for each user
    user_thresholds = (
        final_joined_df.groupBy("user_id")
        .agg(percentile_approx("timestamp", threshold).alias("threshold"))
    )

    # Join the original DataFrame with the user thresholds
    joined_df = final_joined_df.join(user_thresholds, on="user_id")

    # Split the data into train and test based on the threshold
    df_train = joined_df.filter(col("timestamp") <= col("threshold")).drop("threshold")
    df_test = joined_df.filter(col("timestamp") > col("threshold")).drop("threshold")
#     if(drop_low_interactions):
#         unique_users_per_interaction = df_train.groupBy('recording_id').agg(
#             F.countDistinct('user_id').alias('users'))
#         unique_users_per_interaction = unique_users_per_interaction.filter(col('users') > 10)
#         df_train = df_train.join(unique_users_per_interaction.select('recording_id'), on='recording_id') 
        
    return df_train, df_test


def split_on_interaction(final_joined_df, split = 0.8):
    # Split distinct user_ids into train and test sets
    partition = final_joined_df.select('user_id').distinct().randomSplit([split, 1-split], seed=1234)

    train_users = [row.user_id for row in partition[0].collect()]
    test_users = [row.user_id for row in partition[1].collect()]

    # Use DataFrame API to filter the data based on user_ids
    train = final_joined_df.filter(col("user_id").isin(train_users))
    test = final_joined_df.filter(col("user_id").isin(test_users))

    return train, test

def filter_low_interactions(df):
    unique_users_per_interaction=df.groupBy('recording_id').agg(
            F.countDistinct('user_id').alias('users'))
    unique_users_per_interaction = unique_users_per_interaction.filter(col('users') > 20)
    df = df.join(unique_users_per_interaction.select('recording_id'), on='recording_id')
    
    unique_interactions_per_user=df.groupBy('user_id').agg(
            F.countDistinct('recording_id').alias('unique_listens'))
    unique_interactions_per_user=unique_interactions_per_user.filter(col('unique_listens')>20)
    
    df=df.join(unique_interactions_per_user.select('user_id'),on='user_id')
    return df

def GetGroupedByUserItem(df):
    df = df.groupBy("recording_id_index", "user_id").agg(count("*").alias("rating"))
    df_total=df.groupBy("user_id").agg(F.sum("rating").alias("total"))
    #df_total=df_total.select("user_id","total")
    df_joined=df.join(df_total,on='user_id')
    df_joined=df_joined.withColumn("avg_rating",(col("rating")/col("total")))
    return df_joined.select("user_id","avg_rating","recording_id_index")
