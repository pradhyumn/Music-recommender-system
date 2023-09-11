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


def get_stats_df(df, value_col, variable_col):
    return df.select(
        lit(variable_col).alias("Metric"),
        col(f"mean_{value_col}").alias("mean"),
        col(f"std_{value_col}").alias("stddev"),
        col(f"min_{value_col}").alias("min"),
        col(f"max_{value_col}").alias("max")
    )
def getStats(result_df):
    result_df = result_df.cache()

    # Calculate user counts per interaction
    user_counts_per_interaction = result_df.groupBy("recording_id").agg(
        F.count("user_id").alias("user_count"),
        F.countDistinct("user_id").alias("unique_user_count")
    )

    # Calculate interaction counts per user
    interaction_counts_per_user = result_df.groupBy("user_id").agg(
        F.count("recording_id").alias("interaction_count"),
        F.countDistinct("recording_id").alias("unique_interaction_count")
    )

    # Calculate the statistics DataFrames
    users_per_interaction_stats = user_counts_per_interaction.agg(
        F.mean("user_count").alias("mean_user_count"),
        F.stddev("user_count").alias("std_user_count"),
        F.min("user_count").alias("min_user_count"),
        F.max("user_count").alias("max_user_count"),
        F.mean("unique_user_count").alias("mean_unique_user_count"),
        F.stddev("unique_user_count").alias("std_unique_user_count"),
        F.min("unique_user_count").alias("min_unique_user_count"),
        F.max("unique_user_count").alias("max_unique_user_count")
    )
    interactions_per_user_stats = interaction_counts_per_user.agg(
        F.mean("interaction_count").alias("mean_interaction_count"),
        F.stddev("interaction_count").alias("std_interaction_count"),
        F.min("interaction_count").alias("min_interaction_count"),
        F.max("interaction_count").alias("max_interaction_count"),
        F.mean("unique_interaction_count").alias("mean_unique_interaction_count"),
        F.stddev("unique_interaction_count").alias("std_unique_interaction_count"),
        F.min("unique_interaction_count").alias("min_unique_interaction_count"),
        F.max("unique_interaction_count").alias("max_unique_interaction_count")
    )

    # Reformat the statistics DataFrames and combine them
    users_per_interaction_df = get_stats_df(users_per_interaction_stats, "user_count", "users_per_interaction")
    unique_users_per_interaction_df = get_stats_df(users_per_interaction_stats, "unique_user_count", "unique_users_per_interaction")
    interactions_per_user_df = get_stats_df(interactions_per_user_stats, "interaction_count", "interactions_per_user")
    unique_interactions_per_user_df = get_stats_df(interactions_per_user_stats, "unique_interaction_count", "unique_interactions_per_user")

    # Combine the DataFrames
    stats_df = users_per_interaction_df.unionByName(unique_users_per_interaction_df) \
        .unionByName(interactions_per_user_df) \
        .unionByName(unique_interactions_per_user_df)

    return stats_df


def plot_unique_users_per_interaction_histogram_large(df, num_bins=50):
    # Calculate unique user counts per interaction
    user_counts_per_interaction = df.groupBy("recording_id").agg(
        F.countDistinct("user_id").alias("unique_user_count")
    )

    # Determine the range of unique user counts
    min_user_count, max_user_count = user_counts_per_interaction.agg(
        F.min("unique_user_count"), F.max("unique_user_count")
    ).collect()[0]

    # Create the Bucketizer
    bucketizer = Bucketizer(
        splits=[i for i in range(int(min_user_count), int(max_user_count) + 2)], 
        inputCol="unique_user_count", 
        outputCol="buckets"
    )

    # Apply the Bucketizer to the data
    bucketed_data = bucketizer.transform(user_counts_per_interaction)

    # Calculate the histogram data
    histogram_data = bucketed_data.groupBy("buckets").agg(count("*").alias("frequency")).orderBy("buckets").collect()

    # Extract bucket edges and frequencies for plotting
    bucket_edges = [row["buckets"] for row in histogram_data]
    frequencies = [row["frequency"] for row in histogram_data]

    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(bucket_edges, frequencies, width=1, edgecolor='black')
    plt.xlabel('Unique Users per Interaction')
    plt.ylabel('Frequency')
    plt.title('Histogram of Unique Users per Interaction')
#     plt.xscale('log')
    plt.yscale('log')
    plt.show()
