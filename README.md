Spark Python Machine Learning Examples
=====================================

This repository is part of a series on Apache Spark examples, aimed at demonstrating the implementation of Machine Learning solutions in all four different languages supported by Spark. Check the other repositories:

* **Scala**  - [github.com/adornes/spark_scala_ml_examples](https://github.com/adornes/spark_scala_ml_examples)
* **Python** - You are here!
* **Java**   - To be published soon!
* **R**      - To be published soon!

This repository aims at demonstrating how to build a [Spark 2.0](https://spark.apache.org/releases/spark-release-2-0-0.html) application with [Python](http://www.python.org/) for solving Machine Learning problems, ready to be run locally or on any cloud platform such as [AWS Elastic MapReduce (EMR)](https://aws.amazon.com/emr/).

Each Python script in the package can be run as an individual application, as described in the next sections.  

### Why Spark?

Since almost all personal computers nowadays have many Gigabytes of RAM (and it is in an accelerated growing) and powerful CPUs and GPUs, many real-world machine learning problems can be solved with a single computer and frameworks such as [ScikitLearn](http://scikit-learn.org/), with no need of a distributed system, this is, a cluster of many computers. Sometimes, though, data grows and keeps growing. Who never heard the term "Big Data"? When it happens, a non-distributed/scalable solution may solve for a short time, but afterwards such solution will need to be reviewed and maybe significantly changed.

Spark started as a research project at [UC Berkeley](http://www.berkeley.edu/) in the [AMPLab](https://amplab.cs.berkeley.edu/), a research group that focuses on big data analytics. Since then, it became an [Apache](https://www.apache.org/) project and has delivered many new releases, reaching a consistent maturity with a wide range of functionalities. Most of all, Spark can perform data processing over some Gigabytes or hundreds of Petabytes with basically the same programming code, only requiring a proper cluster of machines in the background (check [this link](https://databricks.com/blog/2014/10/10/spark-petabyte-sort.html)). In some very specific cases the developer may need to tune the process by changing granularity of data distribution and other related aspects, but in general there are plenty of providers that automate all this cluster configuration for the developer. For instance, the scripts in this repository used [AWS Elastic MapReduce (EMR)](https://aws.amazon.com/emr/), which plays exactly this role. 


### Why Python?

My preferred option for implementing Spark applications (and applications for other purposes as well) is Scala (as you can read [here](https://github.com/adornes/spark_scala_ml_examples)), however it is a common sense that when the subject is Data Science the competition is really narrowed to [Python](https://www.python.org/) vs [R](https://www.r-project.org/). More in-depth comparisons between these two language can be found [here](http://www.kdnuggets.com/2015/05/r-vs-python-data-science.html) and [here](https://www.datacamp.com/community/tutorials/r-or-python-for-data-analysis#gs.hALDWDM).

[R](https://www.r-project.org/) is the best in terms of libraries for statistical methods, models and graphs. The obvious reason is that it was created (and is maintained) with Statisticians in mind, which also is precisely the same reason why its syntax can look a little bit "unusual" for software engineers. For the same reason, its solid competence goes up to the data analysis output (e.g., reports, slides, output file). There are some libraries for presenting things as an interactive web layer (e.g., [Shiny](https://shiny.rstudio.com/)), but nothing compared to everything available in a general-purpose language such as Python.

So, Python brings with it a wide range of libraries and frameworks for almost everything in the scope of Computer Science and Software Engineering. For Data Analysis, Machine Learning and Data Science as a whole, it has conquered an increasing space with awesome libraries (e.g., [ScikitLearn](http://scikit-learn.org/), [matplotlib](http://matplotlib.org/), [seaborn](http://seaborn.pydata.org/), [pandas](http://pandas.pydata.org/), Spark itself) very competitive with those from R. 

Besides these advantages, Python has always been very successful for its clear syntax and low learning curve. For a team or an individual planning to develop an end-to-end data application, Python is definitely a great choice. For long-term optimization, maintenance and other aspects, a compiled language, such as [Scala](https://www.scala-lang.org), may be a safer and wiser choice.

### Scripts: allstate_claims_severity_GBT_regressor and allstate_claims_severity_random_forest_regressor

[Allstate Corporation](https://www.allstate.com), the second largest insurance company in United States, founded in 1931, recently launched a Machine Learning recruitment challenge in partnership with [Kaggle](https://www.kaggle.com/c/allstate-claims-severity) asking for competitors, Data Science professionals and enthusiasts, to predict the cost, and hence the severity, of claims.
 
The competition organizers provide the competitors with more than 300.000 examples with masked and anonymous data consisting of more than 100 categorical and numerical attributes, thus being compliant with confidentiality constraints and still more than enough for building and evaluating a variety of Machine Learning techniques. 

These two Python scripts obtain the training and test input datasets, from local or [S3](https://aws.amazon.com/s3/details/) environment, and train [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) and [Random Forest](https://en.wikipedia.org/wiki/Random_forest) models over it, respectively.
The objective is to demonstrate the use of [Spark 2.0](https://spark.apache.org/releases/spark-release-2-0-0.html) Machine Learning pipelines with [Python language](http://www.python.org/), [S3](https://aws.amazon.com/s3/details/) integration and some general good practices for building Machine Learning models. In order to keep this main objective, more sophisticated techniques (such as a thorough exploratory data analysis and feature engineering) are intentionally omitted.


#### Flow of Execution and Overall Learnings

Although not so labored in terms of Machine Learning techniques, these scripts provide many important learnings for building ML applications with Spark 2.0, Python and finally running it. Some learnings are detailed as follows:  
 
* Both scripts provide a sophisticated command line interface with [argparse](https://docs.python.org/3/library/argparse.html), through which the runtime can be configured with specific named parameters. It is detailed in the section [Running the Scripts Locally](#running-the-scripts-locally). 
    
* The `process` function is called with an object created by `argparse` which encapsulates the parameters provided at the command line.
    
    ```python
    def process(params):
       ...
    ```

* *SparkSession.builder* is used for building a *Spark session*. It was introduced in Spark 2.0 and is recommended to be used in place of the old *SparkConf* and *SparkContext*. [This link](https://databricks.com/blog/2016/08/15/how-to-use-sparksession-in-apache-spark-2-0.html) provides a good description of this new strategy and the equivalence with the old one.
    
    ```python
    sparkSession = (SparkSession.builder.
      appName("allstate_claims_severity_random_forest_regressor")
      .getOrCreate())
    ```

* The access to S3 is configured with **s3a** support, which compared to the predecessor **s3n** improves the support to large files (no more 5GB limit) and provides higher performance. For more information on this, check [this](https://wiki.apache.org/hadoop/AmazonS3), [this](https://aws.amazon.com/premiumsupport/knowledge-center/emr-file-system-s3/) and [this](http://stackoverflow.com/questions/30385981/how-to-access-s3a-files-from-apache-spark) links.

    ```python
    sparkSession.conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    sparkSession.conf.set("spark.hadoop.fs.s3a.access.key", params.s3AccessKey)
    sparkSession.conf.set("spark.hadoop.fs.s3a.secret.key", params.s3SecretKey)
    ```

* Besides using the new **sparkSession.read.csv** method, the reading process also includes important settings: It is set to read the header of the CSV file, which is directly applied to the columns' names of the dataframe created; and **inferSchema** property is set to *true*. Without the **inferSchema** configuration, the float values would be considered as *strings* which would later cause the **VectorAssembler** to raise an ugly error: `pyspark.sql.utils.IllegalArgumentException: u'Data type StringType is not supported.'`. Finally, both raw dataframes are *cached* since they are again used later in the code for *fitting* the **StringIndexer** transformations and it wouldn't be good to read the CSV files from the filesystem or S3 once again. 

Note: While I was translating this code from the [Scala equivalent](https://github.com/adornes/spark_scala_ml_examples), I forgot the `()` for functions that take no parameters. Later in the code, it would cause the error `Mention AttributeError: 'function' object has no attribute`.

      ```python
      trainInput = (sparkSession.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(params.trainInput)
        .cache())

      testInput = (sparkSession.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(params.testInput)
        .cache())
      ```
  
* The column "loss" is renamed to "label". For some reason, even after using the *setLabelCol* on the regression model, it still looks for a column called "label", raising an ugly error: `pyspark.sql.utils.IllegalArgumentException: u'Field "label" does not exist.'`. It may be hardcoded somewhere in Spark's source code.
 
* The content of *train.csv* is split into *training* and *validation* data, 70% and 30%, respectively. The content of "test.csv" is reserved for building the final CSV file for submission on Kaggle. Both original dataframes are sampled according to command line parameters, which is particularly useful for running fast executions in your local machine;
  
    ```python
    data = (trainInput.withColumnRenamed("loss", "label")
      .sample(false, params.trainSample))
    
    [trainingData, validationData] = data.randomSplit([0.7, 0.3])
    
    trainingData.cache()
    validationData.cache()
    
    testData = testInput.sample(false, params.testSample).cache()
    ```
  
* By using a custom function *isCateg* the column names are filtered and a [StringIndexer](http://spark.apache.org/docs/latest/ml-features.html#stringindexer) is created for each categorical column, aimed at creating a new numerical column according to the custom function *categNewCol*. Note: It is a weak feature engineering, since it is wrong for a learning model to assume that the categories have an order among them (one is greater or less than the other). Whenever categories are confirmed to be unordered, it is better to use some other technique such as [StringIndexer](http://spark.apache.org/docs/latest/ml-features.html#onehotencoder), which yields a different new column for each category holding a boolean (0/1) value;

    ```python
    isCateg     = lambda c: c.startswith("cat")
    categNewCol = lambda c: "idx_{0}".format(c) if (isCateg(c)) else c
    
    stringIndexerStages = map(lambda c: StringIndexer(inputCol=c, outputCol=categNewCol(c))
          .fit(trainInput.select(c).union(testInput.select(c))), filter(isCateg, trainingData.columns))
    ```
  
* There are some very important aspects to be considered when building a feature transformation such as StringIndexer or OneHotEncoder. Such transformations need to be *fitted* before being included in the pipeline and the *fit* process needs to be done over a dataset that contains all possible categories. For instance, if you fit a StringIndexer over the training dataset and afterwards, when the pipeline is used to predict an outcome over another dataset (validation, test, etc.), it faces some unseen category, then it will fail and raise the error: `org.apache.spark.SparkException: Failed to execute user defined function($anonfun$4: (string) => double) ... Caused by: org.apache.spark.SparkException: Unseen label: XYZ ... at org.apache.spark.ml.feature.StringIndexerModel`. This is the reason why the scripts' code fits the StringIndexer transformations over a union of original data from `train.csv` and `test.csv`, bypassing the sampling and split parts.
 
* After the sequence of StringIndexer transformations, the next transformation in the pipeline is the [VectorAssembler](http://spark.apache.org/docs/latest/ml-features.html#vectorassembler), which groups a set of columns into a new "features" column to be considered by the regression model. The filter for only feature columns is performed with the custom function *onlyFeatureCols*. Additionally, the custom function *removeTooManyCategs* is used to filter out some few columns which contain a number of distinct categories much higher than the supported by the default parameter *maxBins* (for RandomForest). In a seriously competitive scenario, it would be better to perform some exploratory analysis to understand these features,  their impact on the outcome variable and which feature engineering techniques could be applied.

    ```python
    removeTooManyCategs = lambda c: not re.match(r"cat(109$|110$|112$|113$|116$)", c)
    
    onlyFeatureCols = lambda c: not re.match(r"id|label", c)
    
    featureCols = map(categNewCol, 
                      filter(onlyFeatureCols, 
                             filter(removeTooManyCategs, 
                                    trainingData.columns)))
    
    assembler = VectorAssembler(inputCols=featureCols, outputCol="features")
    ```
  
* The very last stage in the pipeline is the regression model, which in these scripts is [GBTRegressor](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.GBTRegressor) and [RandomForestRegressor](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.RandomForestRegressor).

    ```python
    algo = RandomForestRegressor(featuresCol="features", labelCol="label")
    
    stages = stringIndexerStages
    stages.append(assembler)
    stages.append(algo)

    pipeline = Pipeline(stages=stages)
    ```
  
* It is interesting to run the pipeline a set of times with different *hyperparameters* for the transformations and the learning algorithm in order to find the combination that best fits the data (see [Hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)). It is also important to evaluate each combination against a separated slice of the data (see [K-fold Cross Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))). For accomplishing such objectives, a [CrossValidator](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) is used in conjunction with a [ParamGridBuilder](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder) (more documentation on (this link)[http://spark.apache.org/docs/latest/ml-tuning.html]) queueing executions with distinct combinations of *hyperparameters* according to which was parametrized in the command line.

    ```python
    paramGrid = (ParamGridBuilder()
      .addGrid(algo.numTrees, params.algoNumTrees)
      .addGrid(algo.maxDepth, params.algoMaxDepth)
      .addGrid(algo.maxBins, params.algoMaxBins)
      .build())
      
    cv = CrossValidator(estimator=pipeline,
                        evaluator=RegressionEvaluator(),
                        estimatorParamMaps=paramGrid,
                        numFolds=params.numFolds)
    
    cvModel = cv.fit(trainingData)
    ```
  
* Note: As observed by [this post](https://databricks.com/blog/2015/01/21/random-forests-and-boosting-in-mllib.html) the Random Forest model is much faster than GBT on Spark. I experienced an execution about 20 times slower with GBT compared to Random Forest with equivalent *hyperparameters*.

* With an instance of [CrossValidatorModel](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidatorModel) already trained, it is time for evaluating the model over the whole training and the validation datasets. From the result of predictions it is possible to easily obtain evaluation metrics with [RegressionMetrics](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RegressionMetrics). Additionally, the instance of the best model can be obtained, providing thus access to some other interesting attributes, such as *featureImportances*.

    ```python
    trainPredictionsAndLabels = cvModel.transform(trainingData).select("label", "prediction").rdd

    validPredictionsAndLabels = cvModel.transform(validationData).select("label", "prediction").rdd

    trainRegressionMetrics = RegressionMetrics(trainPredictionsAndLabels)
    validRegressionMetrics = RegressionMetrics(validPredictionsAndLabels)

    bestModel = cvModel.bestModel
    featureImportances = bestModel.stages[-1].featureImportances.toArray()
    ```
  
* Finally, the model can be used to predict the answer for the *test* dataset and save a csv file ready to be submitted on Kaggle! Again, Spark 2.0 simplifies the process. The function `coalesce` gathers all partitions into 1 only, thus saving a single output file (not many). 
 
    ```python
    cvModel.transform(testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1)
      .write.format("csv")
      .option("header", "true")
      .save(params.outputFile)
    ```
  

#### Running the Scripts Locally

Assuming you have your local environment all set up with Java 8 or higher, Python and Spark 2.0, you can run the desired script (here, allstate_claims_severity_random_forest_regressor) with the following command structure:

```
spark-submit allstate_claims_severity_random_forest_regressor.py --s3AccessKey YOUR_AWS_ACCESS_KEY_HERE --s3SecretKey YOUR_AWS_SECRET_KEY_HERE --trainInput "file:///path/to/the/train.csv" --testInput "file:///path/to/the/test.csv" --outputFile  "file:///path/to/any/name/for/submission.csv" --algoNumTrees 3 --algoMaxDepth 3 --algoMaxBins 32 --numFolds 5 --trainSample 0.01 --testSample 0.01
```

As previously mentioned, [argparse](https://docs.python.org/3/library/argparse.html) is the Python tool that enables the nice names for parameters at command line. If you type something wrong, it will output the sample usage as follows:

```
usage: allstate_claims_severity_random_forest_regressor.py [-h] --s3AccessKey
                                                           S3ACCESSKEY
                                                           --s3SecretKey
                                                           S3SECRETKEY
                                                           --trainInput
                                                           TRAININPUT
                                                           --testInput
                                                           TESTINPUT
                                                           [--outputFile OUTPUTFILE]
                                                           [--algoNumTrees ALGONUMTREES [ALGONUMTREES ...]]
                                                           [--algoMaxDepth ALGOMAXDEPTH [ALGOMAXDEPTH ...]]
                                                           [--algoMaxBins ALGOMAXBINS [ALGOMAXBINS ...]]
                                                           [--numFolds NUMFOLDS]
                                                           [--trainSample TRAINSAMPLE]
                                                           [--testSample TESTSAMPLE]
```

#### Running the Scripts on AWS Elastic MapReduce (EMR)

**EMR** plays the role of abstracting most of the background setup for a cluster with Spark/Hadoop ecosystems. You can actually build as many clusters as you want (and can afford). By the way, the cost for EC2 instances used with EMR is considerably reduced (it is detailed [here](https://aws.amazon.com/emr/pricing)).
 
 Although considerably abstracting the cluster configuration, EMR allows the user to customize almost any of the background details through the *advanced* options of the steps of creating a cluster. For instance, for these Spark scripts, you'll need to customize the Java version, according to [this link](http://docs.aws.amazon.com/ElasticMapReduce/latest/ReleaseGuide/emr-configure-apps.html#configuring-java8). Besides that, everything is created using the options provided. So, going step by step, log in to your AWS console, in the *Services* tab look for *EMR*, select to create a cluster, choose *Go to advanced options* on the top of the screen and fill the options as follows: 
 
 * **Vendor** - Leave it as *Amazon*
 
 * **Release** - Choose *emr-5.1.0*. Select *Hadoop* and *Spark*. I'd also recommend you to select *Zeppelin* (for working with notebooks) and *Ganglia* (for detailed monitoring of your cluster).
 
 * **Edit software settings (optional)** - Ensure the option *Enter configuration* is selected and copy here the configurations of [the aforementioned link](http://docs.aws.amazon.com/ElasticMapReduce/latest/ReleaseGuide/emr-configure-apps.html#configuring-java8)
   
 * **Add steps** - You don't need to do it at this moment. I prefer to do it later, after your cluster is started and ready for processing stuff. Click Next for *Hardware* settings.
 
 * **Hardware** - You can leave it as default (and can also resize it later) but maybe 2 core instances can be increased to 4 or more. Don't forget that your choice will have costs. Click Next for *General Cluster Settings*.
 
 * **Cluster name** - Give some name to your cluster. Feel free to leave all other options with the default values. Click Next for *Security*.
 
 * **EC2 Key Pair** - It is useful if want to log into your EC2 instances via ssh. You can either create a Key Pair or choose some existent if you already have one. Leave the remaining options with the default values and click on *Create Cluster*.
 
 Now you'll have an overview of your cluster's basic data, including the state of your instances. When they indicate to be ready for processing steps, go to the **Steps** tab, click on **Add step** and fill the options as follows:
 
 * **Step type** - Select *Spark application*
  
 * **Application location** - Navigate through your S3 buckets and select the Python script (.py) file there. You'll need to have already uploaded it to S3.
 
 * **Spark-submit options** - It can be blank.
 
 * **Arguments** - Here you type the rest of the command arguments as demonstrated before, but this time indicating S3 paths as follows:
 
 ```
 --s3AccessKey YOUR_AWS_ACCESS_KEY_HERE --s3SecretKey YOUR_AWS_SECRET_KEY_HERE 
 --trainInput "s3:/path/to/the/train.csv" --testInput "s3:/path/to/the/test.csv" 
 --outputFile  "s3:/path/to/any/name/for/submission.csv" 
 --algoNumTrees 20,40,60 --algoMaxDepth 5,7,9 --algoMaxBins 32 --numFolds 10 
 --trainSample 1.0 --testSample 1.0
 ```

That's it! In the list of steps you will see your step running and will also have access to system logs. Detailed logs will be saved to the path defined in your cluster configuration. Additionally, EMR allows the user to clone both steps and clusters, being thus not required to type everything again.


#### Submission on Kaggle

As mentioned along the explanations, many improvements could/should be done in terms of exploratory data analysis, feature engineering, evaluating other models (starting by the simplest ones, as Linear Regression) and then decreasing the predictions error.
 
For being over-simplistic, this model achieved a Mean Absolute Error (MAE) of 1286 in the [public leaderboard](https://www.kaggle.com/c/allstate-claims-severity/leaderboard), far from the top positions.

The submission file and the detailed metrics of the model evaluation can be found under the `output` directory.


### Corrections/Suggestions or just a Hello!

Don't hesitate to contact me directly or create *pull requests* here if you have any correction or suggestion for the code or for this documentation! Thanks! 

* [Github](https://www.github.com/adornes)
* [Twitter](https://twitter.com/daniel_adornes)
* [LinkedIn](https://www.linkedin.com/in/adornes)