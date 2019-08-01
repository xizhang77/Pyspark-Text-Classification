# -*- coding: utf-8 -*-
# Author: Xi Zhang. (xizhang1@cs.stonybrook.edu)

import pandas as pd
import numpy as np

import sys,os,re,glob,math,nltk

from pyspark.sql import SQLContext
from pyspark import SparkContext

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF

###############################################################################
# Importing data and remove unwanted columns
def ImportData():
	sc =SparkContext()
	sqlContext = SQLContext( sc )
	rawData = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('data/train.csv')

	# rawData.printSchema()
	drop_list = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
	data = rawData.select([column for column in rawData.columns if column not in drop_list])

	return data

###############################################################################
# Stemming and Removing the stopwords in each content
def ProcessData( data ):
	# Tokenize the content
	regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="Words", pattern="\\W")
	# regexTokenized = regexTokenizer.transform(data).select("Descript", "Words").show(truncate=False)

	# Remove stopwords
	remover = StopWordsRemover(inputCol="Words", outputCol="Filtered")
	# remover.transform( regexTokenized ).select("Words", "Filtered").show(truncate=False)

	# Encode a string column of labels to a column of label indices
	indexer = StringIndexer(inputCol = "Category", outputCol = "label")

	# Fit the pipeline and generate the final table
	pipeline = Pipeline(stages=[regexTokenizer, remover, indexer])
	
	dataset = pipeline.fit(data).transform(data)
	# dataset.show(5)

	return dataset


###############################################################################
# Getting features for model training
# Both Term Frequency and TF-IDF Score are implemented here
def GetFeatures( data ):
	# Term Frequency
	# minDF: Specifies the minimum number of different documents a term must appear in 
	# 		 to be included in the vocabulary
	# cv = CountVectorizer(inputCol="Filtered", outputCol="features", minDF=2.0)

	# TF-IDF Score
	hashingTF = HashingTF(inputCol="Filtered", outputCol="rawFeatures", numFeatures=10000)
	idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2.0)

	pipeline = Pipeline(stages=[tf, idf])

	dataset = pipeline.fit(data).transform(data)
	dataset.show(5)
	
	return dataset


###############################################################################
# 
def TrainModel( dataset ):
	( trainData, testData ) = dataset.randomSplit([0.7, 0.3], seed = 100)
	# print trainData.count(), testData.count()
	# print trainData.printSchema()

	# Create a LogisticRegression instance. This instance is an Estimator.
	lr = LogisticRegression(maxIter=20, regParam=0.1)
	model = lr.fit( trainData )

	prediction = model.transform( testData )

	# result = prediction.select("Descript", "Category", "Probability", "label", "prediction") \
	# .orderBy("Probability", ascending=False)


	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	print evaluator.evaluate( prediction )


if __name__ == '__main__':
	data = ProcessData( ImportData() )
	dataset = GetFeatures( data )

	TrainModel( dataset )

	
