# Text classfication by PySpark

In this repo, PySpark is used to solve a binary text classification problem. The whole procedure can be find in [Python Code](main.py).

## Data 
Our task here is to general a binary classifier for IMDB movie reviews. The data was collected by Cornell in 2002 and can be downloaded from [Here](http://www.cs.cornell.edu/people/pabo/movie-review-data/).

Before building the models, the raw data (1000 positive and 1000 negative TXT files) is stemmed and integrated into a single CSV file. The code can be find in [Indexing Data](index-data.py). 
