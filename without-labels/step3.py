# -*- coding: utf-8 -*-
# Author: Xi Zhang. (xizhang1@cs.stonybrook.edu)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from os import system

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.feature import HashingTF, IDF

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.sql.types import *

from pyspark import SparkContext

import pandas as pd
from nltk.stem.snowball import SnowballStemmer


###############################################################################
# Importing data 
def ImportData():

	newDF=[StructField('creator_name',StringType(),True),
		StructField('userid',StringType(),True),
		StructField('comment',StringType(),True),
		StructField('label',IntegerType(),True)]
	structure = StructType(fields=newDF)

	df = spark.read.csv('data/step1.csv', header=True, schema=structure)

	return df


###############################################################################
# Stemming and Removing the stopwords in each content
def ProcessData( data ):
	# Tokenize the content
	regexTokenizer = RegexTokenizer(inputCol="comment", outputCol="words", pattern="\\W")

	# Remove stopwords
	remover = StopWordsRemover(inputCol="words", outputCol="filtered")
	# remover.transform( regexTokenized ).select("Words", "Filtered").show(truncate=False)

	# Fit the pipeline and generate the final table
	pipeline = Pipeline(stages=[regexTokenizer, remover])
	
	df = pipeline.fit(data).transform(data)
	# df.printSchema()

	df = df.withColumn('size', F.size(F.col('filtered')) )
	new_df = df.filter(F.col('size') >= 1)

	# new_df.printSchema()
	new_df = new_df.drop('size')

	print("======== Finish Processing Data =========")

	return new_df


###############################################################################
# Getting features for model training
# Both Term Frequency and TF-IDF Score are implemented here
def GetFeatures( data ):
	# Term Frequency
	cv = CountVectorizer(inputCol="filtered", outputCol="features", minDF=0.02, vocabSize=4000)
	model = cv.fit(data)
	df = model.transform(data)

	# print model.vocabulary
	# TF-IDF Score
	# hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=3000)
	# idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2.0)
	# pipeline = Pipeline(stages=[hashingTF, idf])

	# df = pipeline.fit(data).transform(data)

	print("========= Finish Getting Features for Training =========")

	return df


def Prediction( testData, model, flag ):

	prediction = model.transform( testData )
	
	evaluator = BinaryClassificationEvaluator()

	# print evaluator.evaluate( prediction )

	tp = prediction[(prediction.label == flag) & (prediction.prediction == flag)].count()
	fp = prediction[(prediction.label != flag) & (prediction.prediction == flag)].count()

	precision = float(tp) / float(tp + fp)

	print "precision is: ", precision

	
if __name__ == '__main__':
	name = 'step3'

	sc = SparkContext()
	spark = SparkSession.builder.appName(name).getOrCreate()

	data = ProcessData( ImportData() )
	df = GetFeatures( data )

	catmodel = LogisticRegressionModel.load('model/LogisticRegressionModelForCat')
	dogmodel = LogisticRegressionModel.load('model/LogisticRegressionModelForDog')

	Prediction( df, catmodel, 0 )
	Prediction( df, dogmodel, 1 )