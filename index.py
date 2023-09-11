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
from preprocess import GetDataset, preprocess, filter_low_interactions, GetGroupedByUserItem
from baseline_popularity import CalculateGlobalBias, CalculateItemBias, calculate_precision
from time import perf_counter
import argparse



def main(args, spark):
    df_interactions, df_tracks, df_users = GetDataset(spark, version='', data_dir=args.data_dir)
    df_interactions_test, df_tracks_test, df_users_test = GetDataset(spark, version='', split='test',data_dir=args.data_dir)
    df = preprocess(df_tracks, df_interactions)
    df_test = preprocess(df_tracks_test, df_interactions_test)
    df = filter_low_interactions(df)
    df_test = filter_low_interactions(df_test)
    print("Finished Preprocessing.")
    if(args.task == 1):
        global_bias=CalculateGlobalBias(df)
        itemdf = CalculateItemBias(df, global_bias)
        recommendations_final=itemdf.limit(100)
        d_final=calculate_precision(df_test, recommendations_final)
        print("Precision at 100 for Baseline Popularity: ", d_final.precisionAt(100))
        print("meanAveragePrecision at 100 for Baseline Popularity: ", d_final.meanAveragePrecisionAt(100))
        print("meanAveragPrecision for Baseline Popularity: ", d_final.meanAveragePrecision)
        return
    
    combined_df = df.union(df_test)
    indexer = StringIndexer(inputCol="recording_id", outputCol="recording_id_index", handleInvalid="keep")

    # Fit the indexer on your train data and transform both train and test data
    indexer_model = indexer.fit(combined_df)
    # indexer_model = indexer.fit(df_test)
    df_indexed = indexer_model.transform(df)
    df_test_indexed = indexer_model.transform(df_test)
    print("Finished Indexing")
    train_als,test_als = GetGroupedByUserItem(df_indexed),GetGroupedByUserItem(df_test_indexed)
    als = ALS(maxIter=10, regParam=0.01, alpha=0.2, rank=300, userCol="user_id", itemCol="recording_id_index", ratingCol="avg_rating", coldStartStrategy="drop")
    if(args.task == 2):
        pipeline = Pipeline(stages=[als])
        paramGrid = ParamGridBuilder().addGrid(als.rank, [10, 100, 300]) \
                        .addGrid(als.maxIter, [10, 20, 30]) \
                        .addGrid(als.regParam, [0.001,0.01,0.1]) \
                        .addGrid(als.alpha, [0.01, 0.1, 0.5, 1.0]) \
                        .build()
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="avg_rating", predictionCol="prediction")
        crossval = CrossValidator(estimator=pipeline,  # Or use 'als' if you don't have a pipeline
                                estimatorParamMaps=paramGrid,
                                evaluator=evaluator,
                                numFolds=2, parallelism=2)
        cvModel = crossval.fit(train_als)
        predictions = cvModel.transform(test_als)
        rmse = evaluator.evaluate(predictions)
        print("Root-mean-square error = " + str(rmse))
        bestModel = cvModel.bestModel
        bestALSModel = bestModel.stages[-1]  # Or just 'bestModel' if you don't have a pipeline
        print("Best rank:", bestALSModel.rank)
        print("Best maxIter:", bestALSModel._java_obj.parent().getMaxIter())
        print("Best regParam:", bestALSModel._java_obj.parent().getRegParam())
        print("Best alpha:", bestALSModel._java_obj.parent().getAlpha())
        return 
    if(args.task == 3):
        start_time = perf_counter()
        model = als.fit(train_als)
        time_taken = perf_counter() - start_time
        print(f"Time taken to train ALS model: {time_taken} seconds")
        userRecs = model.recommendForAllUsers(100)
        groundTruth = test_als.orderBy.groupBy("user_id").agg(collect_list("recording_id_index").alias("true_positives"))
        userRecs = userRecs.select(col("user_id"), expr("transform(recommendations, x -> x.recording_id_index) as predictions"))
        joinedData = userRecs.join(groundTruth, on="user_id")

        def to_double_list(value):
            return [float(x) for x in value]

        to_double_list_udf = udf(to_double_list, ArrayType(DoubleType()))

        joinedData = joinedData.withColumn("predictions", to_double_list_udf(joinedData["predictions"]))

        evaluator = RankingEvaluator(k=100, metricName="meanAveragePrecision", labelCol="true_positives", predictionCol="predictions")
        map_score = evaluator.evaluate(joinedData)
        print("MAP Score =", map_score)

if __name__ == "__main__":
    # Get arguments from command line using argparser
    task_choices = {1: "Compute Baseline Popularity", 2: "Run crossvalidation hyperparam search for ALS", 3: "Run best param ALS training"}
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/scratch/work/courses/DSGA1004-2021/listenbrainz')
    parser.add_argument("--spark_session_url", type=str, required=True, default="")
    parser.add_argument("--version", type=str, default='')
    # Define the choices for the "task" argument as integers

    # Add the "task" argument with the specified choices and help messages
    parser.add_argument("--task", choices=task_choices.keys(), type=int, default=1, help="Select the task by specifying the corresponding number:\n" +
                                                                                        "1: Compute Baseline Popularity\n" +
                                                                                        "2: Run crossvalidation hyperparam search for ALS\n" +
                                                                                        "3: Run best param ALS training")
    args = parser.parse_args()
    data_dir = args.data_dir
    spark = SparkSession.builder.appName("Group14").master(args.spark_session_url).config("spark.kryoserializer.buffer.max", "1g").getOrCreate()
    sc = spark.sparkContext
    main(args, spark)
