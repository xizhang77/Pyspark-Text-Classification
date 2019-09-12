# Text classfication by PySpark

In this repo, PySpark is used to solve a binary text classification problem. The whole procedure can be find in [Here](main.py).

## Data 
Our task here is to general a binary classifier for IMDB movie reviews. The data was collected by Cornell in 2002 and can be downloaded from [Here](http://www.cs.cornell.edu/people/pabo/movie-review-data/).

Before building the models, the raw data (1000 positive and 1000 negative TXT files) is stemmed and integrated into a single CSV file. The code can be find [Here](index-data.py). 

There are only two columns in the dataset:
* comments: contents in each review
* category: neg/pos.
