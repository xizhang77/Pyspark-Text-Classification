# -*- coding: utf-8 -*-
# Author: Xi Zhang. (xizhang1@cs.stonybrook.edu)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from os import system

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.feature import HashingTF, IDF

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.sql.types import StructField,IntegerType, StructType,StringType


###############################################################################
# Importing data 
def ImportData():
	newDF=[StructField('creator_name',StringType(),True),
		StructField('userid',StringType(),True),
		StructField('comment',StringType(),True),
		StructField('label',IntegerType(),True)]
	structure = StructType(fields=newDF)

	spark = SparkSession.builder.appName(name).getOrCreate()
	df = spark.read.csv('data/step1.csv', header=True, schema=structure)
	
	other_df = df.filter( df.label == 0 )
	owner_df = df.filter( df.label != 0 )
	
	print other_df.count(), owner_df.count()

	return other_df, owner_df


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

	new_df.printSchema()
	new_df.show()


	return new_df


###############################################################################
# Getting features for model training
# Both Term Frequency and TF-IDF Score are implemented here
def GetFeatures( data ):
	# Term Frequency
	# minDF: Specifies the minimum number of different documents a term must appear in 
	# 		 to be included in the vocabulary
	cv = CountVectorizer(inputCol="filtered", outputCol="features", minDF=2.0)
	df = cv.fit(data).transform(data)

	'''
	# TF-IDF Score
	hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
	idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2.0)
	pipeline = Pipeline(stages=[hashingTF, idf])

	df = pipeline.fit(data).transform(data)
	'''

	return df


###############################################################################
# Training logistic regression and get the prediction
def TrainModel( df ):
	print df.count()
	( trainData, testData ) = df.randomSplit([0.7, 0.3], seed = 100)
	# print trainData.count(), testData.count()

	# Create logistic regression
	lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10, regParam=0.1)

	model = lr.fit( trainData )

	prediction = model.transform( testData )

	evaluator = BinaryClassificationEvaluator()

	print evaluator.evaluate( prediction )

	return model 

###############################################################################
# Training logistic regression and get the prediction
def TestModel( testData ):

	prediction = model.transform( testData )

	evaluator = BinaryClassificationEvaluator()

	print evaluator.evaluate( prediction )


if __name__ == '__main__':
	name = 'step2'
	rawdata, ownerdata = ImportData()
	
	# ownerdata = ProcessData( ownerdata )
	# owner_df = GetFeatures( data )

	data = ProcessData( rawdata )
	# df = GetFeatures( data )

	# model = TrainModel( owner_df )