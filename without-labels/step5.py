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
# Get top 10 creators based on fraction of the cat/dog owners
def GetCreator( data ):
	audience = data.groupBy('userid').count()

	total_number = audience.count()
	print total_number

	df = data.filter( data.label != 2 )
	new_df = df.dropDuplicates(['creator_name', 'userid'])

	new_df = data.groupBy('creator_name').count()
	new_df = new_df.withColumn('fraction', F.col('count')/total_number )
	
	new_df.sort(F.col("fraction").desc()).show( 10 )

	
if __name__ == '__main__':
	name = 'step5'

	sc = SparkContext()
	spark = SparkSession.builder.appName(name).getOrCreate()

	data = ImportData()

	GetCreator( data )


	'''
	df = data.groupBy('creator_name').count()

	df.sort(F.col("count").desc()).show( 10 )
	'''