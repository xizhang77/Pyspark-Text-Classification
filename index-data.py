# -*- coding: utf-8 -*-
# Author: Xi Zhang. (xizhang1@cs.stonybrook.edu)

import pandas as pd
import numpy as np

import sys,os,re,glob,nltk


# print "Enter cluster.py"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
os.chdir(STATIC_DIR)

stemmer = nltk.stem.snowball.SnowballStemmer("english")
# stopwords = nltk.corpus.stopwords.words('english')

doc_context = [] 
doc_label = []

for file in glob.glob("*/*"): 
	with open( file, 'r' ) as f:
		context = []
		for line in f.readlines():	
			texts = re.findall(r'[a-zA-Z]+', line) #[\w]: matches any alphanumeric character and the underscore; equivalent to [a-zA-Z0-9_].
			# Stemming the words 
			stemmed = [stemmer.stem(t) for t in texts]
			context +=  stemmed 
		out_str=' '.join(context)
		doc_context.append(out_str)
		doc_label += [ 'pos' ] if file[0] == 'p' else [ 'neg' ]
		# print out_str

dict = {'Comments': doc_context, 'Category': doc_label}

df = pd.DataFrame(dict) 
df.to_csv('comments.csv', header=True, index=False )
