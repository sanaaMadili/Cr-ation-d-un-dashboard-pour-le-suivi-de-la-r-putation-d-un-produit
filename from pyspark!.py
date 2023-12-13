# ... (your existing imports) !pip install pyspark sklearn skl2onnx pandas onnxmltools
""""
!pip install protobuf==3.20.2
!pip install onnx==1.10.1
!pip install onnxmltools==1.9.0
"""

# ... (your existing imports)
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col,when
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.feature import Tokenizer, HashingTF, IDF ,StringIndexer
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import split
from onnxmltools import convert

try:
    spark = SparkSession.builder.appName("SentimentAnalysis").master("local[*]").config("spark.driver.maxResultSize", "8g").config("spark.executor.memory", "6g").getOrCreate()
    print("SparkSession created successfully")

    test_file = "hdfs://namenode:9000/user/input/test.ft.txt"
    train_file = "hdfs://namenode:9000/user/input/train.ft.txt"

    test_df = spark.read.text(test_file)
    train_df = spark.read.text(train_file)

    print("\nFirst few rows of test.ft.txt:")
    #test_df.show(5, truncate=False)

    print("\nFirst few rows of train.ft.txt:")
    #train_df.show(5, truncate=False)
    print("---------------------------__--------------:")
    # drop nan
    test_df = test_df.na.drop()
    train_df = train_df.na.drop()

    test_df.count()
    train_df.count()

    # Train DataFrame
    split_col_tr = split(train_df["value"], " ", 2)
    train_df = train_df.withColumn("label", when(split_col_tr.getItem(0) == "__label__1", 1.0).otherwise(2.0))
    train_df = train_df.withColumn("Comment", split_col_tr.getItem(1))

    # Test DataFrame
    split_col_ts = split(test_df["value"], " ", 2)
    test_df = test_df.withColumn("label", when(split_col_ts.getItem(0) == "__label__1", 1.0).otherwise(2.0))
    test_df = test_df.withColumn("Comment", split_col_ts.getItem(1))

    # Use only the "Comment" column as the feature
    train_processed_df = train_df.select("Comment", "label")
    test_processed_df = test_df.select("Comment", "label")

    print("After preprocessing:")
    train_processed_df.show(4, truncate=False)
    test_processed_df.show(4, truncate=False)

    train_processed_set, val_processed_set = train_processed_df.randomSplit([0.90, 0.10], seed=2000)

    # Continue with the rest of your code for model training and evaluation...

    models = [
        ('NaiveBayes', NaiveBayes()),
        ('RandomForestClassifier', RandomForestClassifier())
    ]

    best_model = None
    best_accuracy = 0.0

    for name, model in models:
        tokenizer = Tokenizer(inputCol="Comment", outputCol="words")
        hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=5000)
        idf = IDF(inputCol="rawFeatures", outputCol="features")

        indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
        classifier = model.setLabelCol("label_indexed").setFeaturesCol("features")

        pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, indexer, classifier])

        print(f"Training {name} model...")
        model_fit = pipeline.fit(train_processed_set)
        print(f"{name} model trained successfully.")

        predictions = model_fit.transform(train_processed_set)
        predictions_val = model_fit.transform(val_processed_set)

        print("predictions-..........")
        #predictions.show(7, truncate=False)
        print("predictions_val-..........")
        #predictions_val.show(7, truncate=False)

        evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print(f'{name} Accuracy: {accuracy}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_fit

    onnx_path = "hdfs://namenode:9000/user/best_model.onnx"
    print("Saving the best model to ONNX format...")
    initial_type = [('string_input', StringType())]
    onnx_bytes = convert.to_onnx(best_model, test_processed_df, initial_types=initial_type)
    with open(onnx_path, "wb") as onnx_file:
        onnx_file.write(onnx_bytes.SerializeToString())

except Exception as e:
    print(f"Error during Spark processing: {e}")
