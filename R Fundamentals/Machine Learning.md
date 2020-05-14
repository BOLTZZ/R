*Note: Check out [data science functions](https://github.com/BOLTZZ/R/blob/master/R%20Fundamentals/R%20Basics%20&%20Syntax.md#import-data-gathering-functions) for data science related functions.*
# Machine Learning:
* Machine learning is based on algorithms built by data. Unlike, Artificial Intelligence (AI) which implements decision making based on programmable rules derived from theory.

<strong>Intro. to Machine Learning:</strong>
* Data comes in the form of the *outcome* we want to predict and the *features* (predictors or covariates) we'll use to predict the outcome. We should build an algorithm that takes feature values as input(s) and returns a prediction for the unkown outcome.
* The machine learning approach is to *train* (trtr) an algorithm using a datasest for which the outcome is known. Then, apply this algorithm when we don't know the outcome.
* Y will denote the outcome and the features will be denoted by X<sub>1</sub>, ..., X<sub>p</sub>.
* Predicton problems can be divided into categorical or continuous outcomes. For categorical, Y can be any of K classes (the number of classes can vary greatly across apliccations). K categories are denoted as: k = 1, ..., K and for binary data it's: k = 0, 1. 
* If we have 5 features and an outcome to predict, it can be drawn as: 
<img src = "https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/features.PNG" width = 700 height = 100>

* To build a model that provides a prediction for a set of values: X<sub>1</sub> = x<sub>1</sub>, ..., X<sub>5</sub> = x<sub>5</sub> we collect data for which we do know the outcome:
<img src = "https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/outcomes.PNG" width = 700 height = 250>

* Ŷ denotes the prediction, actual outcome means observed outcome, and our goal is for Ŷ = actual outcome. The outcome, Y, can be categorical (sex, spam or no spam, and etc) or continuous (movie rating, house price, and more). 
* When the outcome is categorical the machine learning task is referred to as classification. The predictions will be categorical (like outcome) and incorrect or correct. 
* When the outcome is continuous the machine learning task is referred to as prediction. The predictions will not be right or wrong instead, an error will be made. An error is the difference between prediction and actual outcome. 

An example:
* When letters are recieved in the post office, they're sorted by zip code. This is done thanks to machine learning algorithms to read the zip code and robots to sort the letters. 
* The first step in building an algorithm is to identify what are the outcomes and features. *Traning data* is known data that is used to build the algorithm.
* The sent in images are converted to a 28 by 28 pixelated image (784 pixels). For each pixel we obtain a grayscale intensity between 0 (white) and 255 (black), considered continuous. For each digitized image, i, there is a categorical outcome, Y<sub>i</sub>, which can be one of 10 values (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) and there are 784 features (X<sub>i, 1</sub>, ..., X<sub>i, 784</sub>). The bolded <strong>X<sub>i</sub></strong> is used to represent the vector of indivual predictors: <strong>X<sub>i</sub></strong> = X<sub>i, 1</sub>, ..., X<sub>i, 784</sub>. Uppercase values, generally, denote random variables whislt lowercase values, generally, denote observed values (X = x).
