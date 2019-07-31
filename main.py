# -*- coding: utf-8 -*-

from pyspark.sql import SQLContext
from pyspark import SparkContext


sc =SparkContext()
sqlContext = SQLContext(sc)
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('train.csv')