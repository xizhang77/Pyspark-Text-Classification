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
def ProcessData(data):
	# Tokenize the content
	regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="Words", pattern="\\W")
	# regexTokenized = regexTokenizer.transform(data).select("Descript", "Words").show(truncate=False)

	# Remove stopwords
	remover = StopWordsRemover(inputCol="Words", outputCol="Filtered")
	# remover.transform( regexTokenized ).select("Words", "Filtered").show(truncate=False)

	# Extract the vocabulary and generate the token counts
	# minDF: Specifies the minimum number of different documents a term must appear in 
	# 		 to be included in the vocabulary
	cv = CountVectorizer(inputCol="Filtered", outputCol="Features", minDF=2.0)

	# Encode a string column of labels to a column of label indices
	indexer = StringIndexer(inputCol = "Category", outputCol = "Label")

	# Fit the pipeline and generate the final table
	pipeline = Pipeline(stages=[regexTokenizer, remover, cv, indexer])
	
	dataset = pipeline.fit(data).transform(data)
	# dataset.show(5)

	return dataset

###############################################################################
# 
def TrainModel(data):

if __name__ == '__main__':
	data = ImportData()
	dataset = ProcessData( data )

	
