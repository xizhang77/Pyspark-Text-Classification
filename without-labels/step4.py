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



###############################################################################
# Importing data 
def ImportData():
	newDF=[	StructField('features', StringType(),True),
			StructField('weights', DoubleType(),True)]
	structure = StructType(fields=newDF)

	df = spark.read.csv('data/featureMap.csv', header=True, schema=structure)
	
	df.show()

	return df


def Ranking( df ):

	df.sort(F.col("weights").desc()).show( 5 )

	df.sort(F.col("weights")).show( 5 )


	
if __name__ == '__main__':
	name = 'step4'
	spark = SparkSession.builder.appName(name).getOrCreate()

	df = ImportData()

	Ranking( df )

