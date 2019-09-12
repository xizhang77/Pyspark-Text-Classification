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

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


###############################################################################
# Importing data and remove unwanted columns
def ImportData():
	sc =SparkContext()
	sqlContext = SQLContext( sc )
	data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('data/comments.csv')

	# data.printSchema()

	return data

###############################################################################
# Stemming and Removing the stopwords in each content
def ProcessData( data ):
	# Tokenize the content
	regexTokenizer = RegexTokenizer(inputCol="Comments", outputCol="Words", pattern="\\W")
	# regexTokenized = regexTokenizer.transform(data).select("Comments", "Words").show(truncate=False)

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
	'''
	# TF-IDF Score
	hashingTF = HashingTF(inputCol="Filtered", outputCol="rawFeatures", numFeatures=10000)
	idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=4.0)

	pipeline = Pipeline(stages=[hashingTF, idf])

	dataset = pipeline.fit(data).transform(data)
	dataset.show(5)
	'''
	# Term Frequency
	# minDF: Specifies the minimum number of different documents a term must appear in 
	# 		 to be included in the vocabulary
	model = CountVectorizer(inputCol="Filtered", outputCol="features", minDF=0.02).fit(data)
	df = model.transform(data)

	print("========= Finish Getting Features for Training =========")

	return df, model.vocabulary


###############################################################################
# Training logistic regression and get the prediction
def TrainModel( df ):
	( trainData, testData ) = df.randomSplit([0.8, 0.2], seed=100)
	# print trainData.count(), testData.count()
	# print trainData.printSchema()

	# Create a LogisticRegression instance. This instance is an Estimator.
	lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0)
	
	lrmodel = lr.fit( trainData )
	prediction = lrmodel.transform( testData )

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")

	# The improvement by Cross Validation is not significant. 
	# Therefore, using the basic model to improve the running time
	'''
	# Create ParamGrid for Cross Validation
	

	paramGrid = (ParamGridBuilder()
		.addGrid(lr.regParam, [0.01, 0.5, 2.0])
		.addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
		.addGrid(lr.maxIter, [1, 5, 10])
		.build())
	cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

	cvmodel = cv.fit( trainData )

	prediction = cvmodel.transform( testData )
	'''

	# result = prediction.select("Category", "Probability", "label", "prediction") \
	# .orderBy("Probability", ascending=False)

	print 'The accuacy of classifier is:', evaluator.evaluate( prediction )

	return lrmodel

###############################################################################
# Getting the feature-weight map
def FeatureMap( vocabulary, coefficients ):
	feature_weight = []
	for i in range( len(coefficients) ):
		feature_weight.append( [vocabulary[i], coefficients[i]])

	weights = pd.DataFrame(feature_weight, columns =['Words', 'Weights'])

	print("========= Finish Getting Feature Maps =========")

	print weights.sort_values(by = 'Weights')


if __name__ == '__main__':
	data = ProcessData( ImportData() )
	df, vocabulary = GetFeatures( data )

	lrmodel = TrainModel( df )

	FeatureMap( vocabulary, lrmodel.coefficients )
