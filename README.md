# Text classfication by PySpark

In this repo, PySpark is used to solve a binary text classification problem. The whole procedure can be find in [Python Code](main.py).

The files in /flask and /lib folders are Flask framework and extensions that will be used for the application. All the executable files are in the /application folder.

* Data processing is done in *cluster.py* file
* The Flask application object creation is in the *\__init\__.py* file
* All the view functions are in the views.py file and imported in the *\__init\__.py* file.
* D3 code could be found in *templates/post.html*



## Data 
Our task is to classify San Francisco Crime Description into 33 pre-defined categories. The data can be downloaded from [Here](http://www.cs.cornell.edu/people/pabo/movie-review-data/).
