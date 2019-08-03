# -*- coding: utf-8 -*-
# Author: Xi Zhang. (xizhang1@cs.stonybrook.edu)

from pyspark.sql import SQLContext
from pyspark import SparkContext

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from os import system


###############################################################################
# Importing data 
def ImportData():
	spark = SparkSession.builder.appName(name).getOrCreate()
	df = spark.read.csv('data/animals_comments.csv', header=True)
	df = df.dropDuplicates()

	df.printSchema()

	return df

###############################################################################
# Filtering data and get the label
def FilterData( df ):
	# First filter: owner?
	# Second filter: cat/dog owner?
	filter_list1 = ['have', 'had', 'got', 'own', 'get', 'adopt', 'adopted']
	filter_list_cat = ['cat', 'kitten']
	filter_list_dog = ['dog', 'puppy']


	# Mark the owner
	df_owner = df.withColumn(
		'catowner', 
		F.col('comment').rlike('(^|\s)(' + '|'.join(filter_list1) + ')(^|\s)(' + '|'.join(filter_list_cat) + ')(\s|$)')
		)
	df_owner = df_owner.withColumn(
		'dogowner', 
		F.col('comment').rlike('(^|\s)(' + '|'.join(filter_list1) + ')(^|\s)(' + '|'.join(filter_list_dog) + ')(\s|$)')
		)
	
	# Get the userid for all the cat/dog owners
	df_catowner = df_owner.filter( df_owner.catowner == True )
	df_dogowner = df_owner.filter( df_owner.dogowner == True )

	userid_list_cat = [row.userid for row in df_catowner.select('userid').collect()]
	userid_list_cat = list( set(userid_list_cat) )

	userid_list_dog = [row.userid for row in df_dogowner.select('userid').collect()]
	userid_list_dog = list( set(userid_list_dog) )

	# Generate labels for cat/dog owners
	df_owner = df_owner.withColumn(
		'label',
		F.when(F.col('userid').isin(userid_list_cat), 1) \
		.when(F.col('userid').isin(userid_list_dog), 2) \
		.otherwise(0)
	)

	columns_to_drop = ['catowner', 'dogowner']
	df_with_label = df_owner.drop(*columns_to_drop)


	df_with_label.printSchema()

	return df_with_label



if __name__ == '__main__':
	name = 'step1'
	rawdata = ImportData()
	df = FilterData( rawdata )

	(
		df
		.repartition(1)
		.write.csv( name + '.csv', header=True )
		)