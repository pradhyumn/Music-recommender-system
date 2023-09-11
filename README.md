# Overview

Recommendation systems are essential for delivering personalized song suggestions on platforms like Spotify and YouTube Music. In applications like Music retrieval system, where recommendations play a big role, it is important to use enhanced content-based similarity by learning from a subset of collaborative filter data, addressing the limitations of collaborative filtering in music recommendation tasks, particularly for new or less popular items. Various algorithms underpin recommendation systems, including popularity-based recommendations, collaborative filtering, and content-based filtering. Popularity-based recommendations emphasize an item's popularity, such as the number of listens a song receives or the unique users that have engaged with it. In this report, we explore our implementation of a popularity-based baseline model, a latent-factor model using Alternating-Least Squares (ALS) for song recommendations, utilizing the ListenBrainz dataset to create a tailored recommender system. We also implement a single-machine implementation of the LightFM model and benchmark the comparison of the training time and precision accuracy for the ALS and LightFM implementations.

## The data set

In this project, we'll use the [ListenBrainz](https://listenbrainz.org/) dataset, which is for use in Google Dataproc's HDFS or any other HPC cluster as well.

This data consists of *implicit feedback* from music listening behavior, spanning several thousand users and tens of millions of songs.
Each observation consists of a single interaction between a user and a song.
**Note**: this is real data.  It may contain offensive language (e.g. in song titles or artist names).  It is entirely possible to complete the assignment using only the interaction data and ID fields without investigating metadata.


## Basic recommender system

1.  As a first step, we partition the interaction data into training and validation samples using the ```preprocess.py``` script.

2.  Before implementing a sophisticated model, we begin with a popularity baseline model. The code for this is in the ```baseline_popularity.py``` script.

3.  Our model uses Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.
    Be sure to thoroughly read through the documentation on the [pyspark.ml.recommendation module](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html) before getting started.
    This model has some hyper-parameters that we tune to optimize performance on the validation set using grid search methods, notably: 
      - the *rank* (dimension) of the latent factors,
      - implicit feedback parameter (alpha),
      - the regularization parameter.

### Evaluation

Once you are able to make predictions—either from the popularity baseline or the latent factor model—you will need to evaluate accuracy on the validation and test data.
Scores for validation and test are reported in the write-up.
Evaluations should be based on predictions of the top 100 items for each user, and report the ranking metrics provided by spark.
Refer to the [ranking metrics](https://spark.apache.org/docs/3.0.1/mllib-evaluation-metrics.html#ranking-systems) section of the Spark documentation for more details.

The choice of evaluation criteria for hyper-parameter tuning is up to you, as is the range of hyper-parameters you consider, but be sure to document your choices in the final report.
As a general rule, you should explore ranges of each hyper-parameter that are sufficiently large to produce observable differences in your evaluation score on the validation data.
If your selection is picking the largest or smallest setting of a hyper-parameter from your range, this probably means your range isn't large enough.

### Documentation

Please refer to the final report (Recommender_System.pdf) document for a detailed explanation of the approach as well as the results.

### Single-Machine Implementation

  - *Comparison to single-machine implementations*: compare Spark's parallel ALS model to a single-machine implementation, e.g. [lightfm](https://github.com/lyst/lightfm) or [lenskit](https://github.com/lenskit/lkpy).  Our comparison should measure both efficiency (model fitting time as a function of data set size) and resulting accuracy.

### Results

- The results for the popularity-based model are:
```
precision@100 - 0.05
mAP Score - 0.16
```
- The results for the ALS-based Latent factor model are:
```
precision@100 - 0.22
mAP Score - 0.25
```
- The results for LightFM (single-machine implementation) model are:
```
precision@100 - 0.14
```
- However, the LightFM version could converge in a 10-times less time than the ALS-based latent factor model, greatly reducing the training time for the recommender system.
  
