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


def CalculateGlobalBias(df_train, beta = 100000):
    total_interactions = df_train.count()
    unique_interactions = df_train.agg(F.countDistinct('recording_id').alias('count')).collect()[0]['count']
    return total_interactions/(unique_interactions + beta)

def CalculateItemBias(df_train, global_bias, beta=5000):
    interactions_by_users = (
        df_train.groupBy("recording_id")
        .agg(
            F.count("user_id").alias("total_play_count"),
            F.countDistinct("user_id").alias("unique_user_count"),
        )
    )

    interactions_by_users = interactions_by_users.withColumn(
        "naive_score", (col("total_play_count") - global_bias) / (col("unique_user_count") + beta)
    )
    song_metrics=interactions_by_users.orderBy(F.desc("naive_score"))
    #Normalize the metrics
    minmaxusers = song_metrics.agg(
        F.min('unique_user_count').alias('min'),
        F.max('unique_user_count').alias('max')).collect()[0]
    min_unique_users, max_unique_users = minmaxusers.min, minmaxusers.max
    minmaxscore = song_metrics.agg(
        F.min('naive_score').alias('min'),
        F.max('naive_score').alias('max')).collect()[0]
    minscore, maxscore = minmaxscore.min, minmaxscore.max
    
    song_metrics = song_metrics.withColumn("normalized_unique_users", (F.col("unique_user_count") - min_unique_users) / (max_unique_users- min_unique_users))
    song_metrics = song_metrics.withColumn("normalized_score", (F.col("naive_score") - minscore) / (maxscore - minscore))
    
    # Calculate a combined score
    user_weight = 0.5 # change this to reflect how important you consider the number of unique users
    play_weight = 0.5  # change this to reflect how important you consider the total play count

    song_metrics = song_metrics.withColumn("combined_score",
                                           user_weight * F.col("normalized_unique_users") + 
                                           play_weight * F.col("normalized_score"))
    
    song_metrics=song_metrics.select("recording_id","combined_score")

    return song_metrics.orderBy(F.desc("combined_score"))

from pyspark.mllib.evaluation import RankingMetrics
def calculate_precision(df_test,recommendations):
    #recommendations = recommendations.select('recording_id').limit(100)
    user_songs = df_test.rdd.map(lambda row: (row.user_id, row.recording_id)).groupByKey().mapValues(list)
    top_songs = recommendations.select('recording_id').rdd.flatMap(lambda x: x).collect()
    ranking_rdd = user_songs.map(lambda x: (x[1], top_songs))
    metrics = RankingMetrics(ranking_rdd)
    return metrics
#     map_score = metrics.meanAveragePrecisionAt(100)
#     print("Mean Average Precision = ", map_score)