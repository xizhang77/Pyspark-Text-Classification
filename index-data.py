# -*- coding: utf-8 -*-
# Author: Xi Zhang. (xizhang1@cs.stonybrook.edu)

import pandas as pd
import numpy as np

import sys,os,re,glob,nltk


# print "Enter cluster.py"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
os.chdir(STATIC_DIR)

stopwords = nltk.corpus.stopwords.words('english')


doc_context = [] 
doc_label = []

for file in glob.glob("*/*"): 
	with open( file, 'r' ) as f:
		context = []
		for line in f.readlines():	
			match = re.findall(r'[a-zA-Z]+', line) #[\w]: matches any alphanumeric character and the underscore; equivalent to [a-zA-Z0-9_].
			context +=  match 
		# print lines_list
		texts = [word for word in context if word not in stopwords]
		out_str=' '.join(texts)
		doc_context.append(out_str)
		doc_label += [ 1 ] if file[0] == 'p' else [ 0 ]
		print out_str



