[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/XGn1nos4)
# DSGA1004 - BIG DATA
## Final project

*Handout date*: 2023-04-12

*Checkpoint submission*: 2023-04-28

*Submission deadline*: 2023-05-16


# Overview

In the final project, you will apply the tools you have learned in this class to solve a realistic, large-scale applied problem.
Specifically, you will build and evaluate a collaborative-filter based recommender system. 

In either case, you are encouraged to work in **groups of up to 3 students**:

- Groups of 1--2 will need to implement one extension (described below) over the baseline project for full credit.
- Groups of 3 will need to implement two extensions for full credit.

## The data set

In this project, we'll use the [ListenBrainz](https://listenbrainz.org/) dataset, which we have prepared for you in Dataproc's HDFS at `/user/bm106_nyu_edu/1004-project-2023/`.  A copy is also available on the Greene cluster at `/scratch/work/courses/DSGA1004-2021/listenbrainz/`.

This data consists of *implicit feedback* from music listening behavior, spanning several thousand users and tens of millions of songs.
Each observation consists of a single interaction between a user and a song.
**Note**: this is real data.  It may contain offensive language (e.g. in song titles or artist names).  It is entirely possible to complete the assignment using only the interaction data and ID fields without investigating metadata.


## Basic recommender system [80% of grade]

1.  As a first step, you will need to partition the interaction data into training and validation samples as discussed in lecture.
    I recommend writing a script do this in advance, and saving the partitioned data for future use.
    This will reduce the complexity of your experiment code down the line, and make it easier to generate alternative splits if you want to measure the stability of your implementation.
    You will also need to aggregate the individually observed interaction events into count data.

2.  Before implementing a sophisticated model, you begin with a popularity baseline model as discussed in class.
    This should be simple enough to implement with some basic dataframe computations, and should be optimized to perform as well as possible on your validation set.
    Evaluate your popularity baseline (see below) before moving on to the next step.

3.  Your recommendation model should use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.
    Be sure to thoroughly read through the documentation on the [pyspark.ml.recommendation module](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html) before getting started.
    This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably: 
      - the *rank* (dimension) of the latent factors,
      - implicit feedback parameter (alpha),
      - the regularization parameter.

### Evaluation

Once you are able to make predictions—either from the popularity baseline or the latent factor model—you will need to evaluate accuracy on the validation and test data.
Scores for validation and test should both be reported in your write-up.
Evaluations should be based on predictions of the top 100 items for each user, and report the ranking metrics provided by spark.
Refer to the [ranking metrics](https://spark.apache.org/docs/3.0.1/mllib-evaluation-metrics.html#ranking-systems) section of the Spark documentation for more details.

The choice of evaluation criteria for hyper-parameter tuning is up to you, as is the range of hyper-parameters you consider, but be sure to document your choices in the final report.
As a general rule, you should explore ranges of each hyper-parameter that are sufficiently large to produce observable differences in your evaluation score on the validation data.
If your selection is picking the largest or smallest setting of a hyper-parameter from your range, this probably means your range isn't large enough.

If you like, you may also use additional software implementations of recommendation or ranking metric evaluations, but be sure to cite any additional software you use in the project.


### Using the cluster

Please be considerate of your fellow classmates!
The Dataproc cluster is a limited, shared resource. 
Make sure that your code is properly implemented and works efficiently. 
If too many people run inefficient code simultaneously, it can slow down the entire cluster for everyone.

**NOTE**: At the time the project is released (2023-04-12), the version of Parquet currently installed on Dataproc has a known issue with reading large parquet files.  It should not cause any problems here, but just in case it does, we have provided a reduced training set (`*_train_small.parquet`) for you to work with.


## Extensions [20% of grade]

For full credit, implement an extension on top of the baseline collaborative filter model.
Again, if you're working in a group of 3, you must implement two extensions for full credit.

The choice of extension is up to you, but here are some ideas:

  - *Comparison to single-machine implementations*: compare Spark's parallel ALS model to a single-machine implementation, e.g. [lightfm](https://github.com/lyst/lightfm) or [lenskit](https://github.com/lenskit/lkpy).  Your comparison should measure both efficiency (model fitting time as a function of data set size) and resulting accuracy.
  - *Fast search*: use a spatial data structure (e.g., LSH or partition trees) to implement accelerated search at query time.  For this, it is best to use an existing library such as [annoy](https://github.com/spotify/annoy), [nmslib](https://github.com/nmslib/nmslib), or [scann](https://github.com/google-research/google-research/tree/master/scann) and you will need to export the model parameters from Spark to work in your chosen environment.  For full credit, you should provide a thorough evaluation of the efficiency gains provided by your spatial data structure over a brute-force search method, as well as any changes in accuracy induced by the approximate search.
  
Other extension ideas are welcome as well, but must be approved in advance by the instructional staff.

## What to turn in

In addition to all of your code, produce a final report (no more than 5 pages), describing your implementation, evaluation results, and extensions.
Your report should clearly identify the contributions of each member of your group. 
If any additional software components were required in your project, your choices should be described and well motivated here.  

Include a PDF of your final report through Brightspace.  Specifically, your final report should include the following details:

- Link to your group's GitHub repository
- Documentation of how your train/validation splits were generated
    - Any additional pre-processing of the data that you decide to implement
- Evaluation of popularity baseline
- Documentation of latent factor model's hyper-parameters and validation
- Evaluation of latent factor model
- Documentation of extension(s)

Any additional software components that you use should be cited and documented with installation instructions.

## Timeline

It will be helpful to commit your work in progress to the repository.
Toward this end, we recommend the following timeline to stay on track:

- [ ] 2023/04/21: data pre-processing, train/validation partitioning, popularity baseline model.
- [ ] **2023/04/28**: evaluation, checkpoint submission with baseline results.
- [ ] 2023/05/05: Working latent factor model implementation on subsample of training data.
- [ ] 2023/05/12: Scale up to the full dataset and develop extensions.
- [ ] 2023/05/16: final project submission.  **NO EXTENSIONS PAST THIS DATE.**
