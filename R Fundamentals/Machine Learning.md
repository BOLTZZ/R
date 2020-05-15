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

<strong>Machine Learning Basics:</strong>

Maching Learning Algorithms Basics:
* The caret package ([cheatsheet](https://d33wubrfki0l68.cloudfront.net/ad16acdb544c1a9ca00c7dd175312a52f45e8979/7e9a2/wp-content/uploads/2015/01/caret-cheatsheet.png)) has a lot of functions useful for building and assembling machine learning algorithms.
* We're going to try to predict sex based on height:
  * For this example, the heights dataset in the dslabs package will be used. The outcome is sex (Y) and the predictor is height (X), the outcome is categorical since Y can be male or female. We won't be able to predict Y (sex) very accuratley based on X (height) since male and female heights aren't that different relative to group variability. 
  * In machine learning, to test the algorithm we split the dataset we're using into 2 and act as if we don't know the outcome for 1 of these 2 sets. Since, the end algorithm will be used by user's whose data isn't in our datset we used to build the algorithm. The group which we used to develop the algorithm is known as the *training set* and the group for which we pretend we don't know the algorithm as the *test set*. 
  * One way to accquire the test and training sets is to randomly split up the data. The caret package has a function for this called createDataPartition(): 
```r
set.seed(3)
test_index = createDataParition(y_value_outcome_value_, times = 1, p = 0.5, list = FALSE) # arg times = how many random samples of indexes to return; arg p = proportion of index represented; arg list = indexes returnes as a list or not.
train_set = heights[-test_index, ]
test_set = heights[test_index, ]
# The sets are defined up above.
```
* Example cont:
  * Now, the algorithm will be developed only using the training set. Then it'll be *freezed* and evaluated using the teset set. The simplest way to evaluate the algorithm when the outcomes are categorical is to report the proportion of cases that were correct in the test set, referred to as *overall accuracy*.
  * It's reccomended that categorical outcomes in machine learning be coded as factors: ```factor(levels = levels(test_set$sex))```.
  * Exploratory data suggests that males, on average, are slightly taller than females which we can use for our machine learning algorithm. So we create a cutoff height for females, so your sex is male if your taller than the cutoff. We examine a bunch of different cutoffs, ranging from 61 inches to 70 inches, and find which one has the best accuracy in the training set.
  * We find the cutoff of 64 inches gives the best accuracy, at ~ 83%. Just testing the algorithm on the training set can lead to *overfitting* which is an overly optimistic evaluation. So, we test our algorithm on the test set and get an accuracy of ~ 81% (which is a difference of ~ 2% from the training set), so our evaluation isn't too optimistic. 
  * Final code:
```r
# Get the libraries and data loaded:
library(tidyverse)
library(caret)
library(dslabs)
data(heights)
# define the outcome and predictors
y <- heights$sex
x <- heights$height
# generate training and test sets
set.seed(2007)
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
test_set <- heights[test_index, ]
train_set <- heights[-test_index, ]
# examine the accuracy of 10 cutoffs
cutoff <- seq(61, 70)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% 
    factor(levels = levels(test_set$sex))
  mean(y_hat == train_set$sex)
})
# Find the best cutoff and run the algorithm on the test data:
max(accuracy)
best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff
y_hat <- ifelse(test_set$height > best_cutoff, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
y_hat <- factor(y_hat)
mean(y_hat == test_set$sex)
```
* Overall Accuracy for 2 features (Petal width and length) in a different datset:
```r
# Load libraries and set everything up:
library(caret)
data(iris)
iris <- iris[-which(iris$Species=='setosa'),]
y <- iris$Species
# Plot a graph of different features (Sepal length and width, Petal length and width) combinations to see what features combinations could be good.
plot(iris,pch=21,bg=iris$Species)
# From the plot Petal.Length and Petal.Width look like a good combination.
# Create testing and training data:
set.seed(2)
test_index <- createDataPartition(y,times=1,p=0.5,list=FALSE)
test <- iris[test_index,]
train <- iris[-test_index,]
# Generate a sequence of cutoffs for petal length and width:            
petalLengthRange <- seq(range(train$Petal.Length)[1],range(train$Petal.Length)[2],by=0.1)
petalWidthRange <- seq(range(train$Petal.Width)[1],range(train$Petal.Width)[2],by=0.1)
# Create a function to find the best cutoff for petal length:
length_predictions <- sapply(petalLengthRange,function(i){
		y_hat <- ifelse(train$Petal.Length>i,'virginica','versicolor')
		mean(y_hat==train$Species)
	})
length_cutoff <- petalLengthRange[which.max(length_predictions)] # 4.7
# Create a function to find the best cutoff for petal width
width_predictions <- sapply(petalWidthRange,function(i){
		y_hat <- ifelse(train$Petal.Width>i,'virginica','versicolor')
		mean(y_hat==train$Species)
	})
width_cutoff <- petalWidthRange[which.max(width_predictions)] # 1.5
# Find the overall accuracy of either greater than length cutoff OR width cutoff. Acuraccy = 88%
y_hat <- ifelse(test$Petal.Length>length_cutoff | test$Petal.Width>width_cutoff,'virginica','versicolor')
mean(y_hat==test$Species)
# Find the overall accuracy of greater than length cutoff AND widht cutoff. Acuraccy = 92%
y_hat <- ifelse(test$Petal.Length>length_cutoff & test$Petal.Width>width_cutoff,'virginica','versicolor')
mean(y_hat==test$Species)
```
Confusion Matrix:
* We used a cutoff of 64 inches but the average female if 65 inches tall, so this prediction rule seems wrong. But, generally, overall accuracy can be a deceptive measure. To see this we'll create a *confusion matrix*, which tabulates each combination of prediction and actual value.
* The confusion matrix can be made using the function table(): ```table(predicted = y_hat, actual = test_set$sex)```. It acutally reveals a problem, computing the accuracy per sex:
```r
test_set %>% 
  mutate(y_hat = y_hat) %>%
  group_by(sex) %>% 
  summarize(accuracy = mean(y_hat == sex))
```
Reveals that there's a very high accuracy for males (~93%) and a very low accuracy for females (~42%). There's an imbalance in accuracy, more than 50% of females are predicted to be males. The reason the overall accuracy is ~ 83% is because of the *prevelance*, there are more males in the dataset than females. In fact, our dataset has 77% males. The large mitakes for females is outweighed by the gains in correct calls for males.
* This bias in the dataset can be a big problem in machine learning. If the traning data is biased the algorithm is likely to be biased as well. The test set is also affected because it was derived from the bias dataset. There are serveral ways we can rid of the bias or make sure prevelance doesn't cloud our assesments via the confusion matrix.
* A general improvment over using overall acurracy is to study sensitivity and specificity seperatley. To define sensitivity and specificity we need a binary outcome. Wehn the outcomes are categorical qe can define these terms for a specific category. Like, in the digits example (reading digits 0 - 9) we can ask for the specificty in the case of correctly predicting 2 as opposed to some other digit.
* Once we specify a category of intereset then we can think about positive outcomes (Y = 1) and negative outcomes (Y = 0). 
* *Sensitivity* is defined as the ability of an algorithm to predict a positive outcome when the actual outcome is positive (Ŷ = 1 when Y = 1). Since an algorithm that predicts a positive outcome (Y = 1) no matter what has perfect sensitivity, the metric on its own is not good enough to judge an algorithm. High sensitivity: Y = 1 -> Ŷ = 1. Also, sensitivity() can be used to find the sensitivity of a prediction: ```sensitivity(predicted_outcome_data, actual_data)```.
* *Specificity* is the ability of an algorithm to predict a negative when the observed outcome is negative (Ŷ = 0 when Y = 0). High specificity: Y = 0 -> Ŷ = 0. Another way to define specificity is the portion of positive calls that're actually positive. In this case, High specificity: Ŷ = 1 -> Y = 1. Also, specificity() can be used to find the specificity of a prediction: ```specificity(predicted_outcome_data, actual_data)```.
* To provide a precise defintion the 4 entries of the confusion matrix are labeled:

| | Actually Positive | Actually Negative
|--|--|--
|Predicted Positive|	True positives (TP)|	False positives (FP)
|Predicted Negative|	False negatives (FN)|	True negatives (TN)
* Sensitivity = TP / (TP + FN) or the proportion of True Positives in the Actually Positive column, this quantitiy is referred to as the *true positive rate (TPR) or recall*. Specificity = TN / (TN + FP) or the proportion of True Negatives in the Actually Negative column, this quantatity is referred to as the *true negative rate (TNR)*. Another way of quantifying specificity = TP / (TP + FP) or the proportion of True Positives in the Predicted Positive row, this quantatity is referred to as the *positive predicted value (PPV) or precision*.
* Unlike the TPR or TNR the PPV depends on prevelance since higher prevelance implies you can get higher PPV (precision) even when guessing. 

|A measure of | Name 1 | Name 2 | Definition | Probability Representation | 
| -- | -- | -- | -- | -- |
| Sensitivity | True Positive Rate (TPR) | Recall | TP / (TP + FN) | Pr(Ŷ = 1 | Y = 1) | 
| Specificity | True Negative Rate (TNR) | 1 - False Positive Rate (FPR) | TN / (TN + FP) | PR(Ŷ = 0 | Y = 0)
| Specificty | Positive Predicted Value (PPV) | Precision | TP / (TP + FP) | PR(Y = 1 | Ŷ = 1)
* The confusionMatrix() in caret computes all the above metrics once a positive is defined: ```confusionMatrix(data = y_hat, reference = test_set$sex) # The function expects factors as inputs and the 1st level is considered the positive outcome or y = 1. In this case, female is positive since it comes before male alphabetically.```
* Even though using specificity and sensitivity is encouraged, it useful to have a 1 number summary (like optimization purposes). One metric, preferred over overall accuarcy is the average of specificity and sensitivity, referred to as the *balance accuracy*. Since specificity and sensitivity are rates it's better to compute the *harmonic mean* (*F<sub>1</sub> score*) like this: 1/0.5 * (1/recall + 1/precision) or written as: 2 * ((precision * recall)/(precision + recall)).
* In different types of contexts, some types of errors are more costly than others. For example, in the case of plane safety, it's much more important to maximize sensitivity over specificity. Failing to predict a plane will malfunction before it crashes is a much more costly error than grounding the plane, when it's actually in perfect condition. On the other hand, in a capital murder case the opposite is true since a false positive can lead to killing an innocent person. 
* The F<sub>1</sub> score can be adapted to weigh specificity and sensitivity differently. To do this we define β to define how much more important specificity is compared to sensitivity and then use a weight harmonic average: 1/(((β<sup>2</sup>/1 + β<sup>2</sup>) * (1/recall)) + ((1/1 + β<sup>2</sup>) * (1/precision))). 
* The F_meas() function in caret computes the summary with β defaulting to 1: ```F_meas(data = y_hat, reference = factor(train_set$sex)) #Since beta defaults to 1 we don't need to change the value of beta.```. We can incorporate this into the code and find the best cutoff:
```r
# maximize F-score
cutoff <- seq(61, 70)
F_1 <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% 
    factor(levels = levels(test_set$sex))
  F_meas(data = y_hat, reference = factor(train_set$sex))
})
max(F_1)

best_cutoff <- cutoff[which.max(F_1)]
y_hat <- ifelse(test_set$height > best_cutoff, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
sensitivity(data = y_hat, reference = test_set$sex)
specificity(data = y_hat, reference = test_set$sex)
```
This results in a cutoff of 66 inches with a percentage of 61%. The 66 inches cutoff makes much more sense than 64 inches and balances the specificity and sensitivity in the confusion matrix.
* A machine learning algorithm with very high specificity and sensitivity may not be useful in practice when prevelance is closer to either 0 or 1. An example of this would be a doctor who specalizes in a rare disease and is intrested in developing an algorithm to see who has the disease. You develop an algorithm with very high sensitivity because if the patient has the disease the algorithm is very likely to predict correctly. But, because of high sensitivity there will be a lot of false positives, in fact, half of your testing data results in positives even though the prevelance of the disease is 5 in 1,000. And, to correctly diagnose the precision matters the most.
* We can calculate the precision of your algorithm to correctly diagnose the disease, Pr(Y = 1 | Ŷ = 1). Using Bayes theorem the 2 measures can be connected, resulting in: Pr(Y | Ŷ = 1) = Pr(Y = 1 | Ŷ = 1) * Pr(Y = 1)/Pr(Ŷ = 1). And, we already know your dataset has a prevelance of 50% for the disease while the actual rate is 0.5%. This results in: Pr(Y = 1)/Pr(Ŷ = 1) = 50%/0.5% = 0.01, which means your precision is less than 0.01.
* One way to compare sensitivty and specificity is to graph them and check for bias. One way to do this is by using the *reciever operating characteristic (ROC) curve*. The ROC curve is the sensitivity (TPR) versus specificity (1 - FPR). We can construct a ROC curve to find the if the height based cutoff method is good or the guessing sex method is good (a perfect method would shoot straight up to 1 and then stay there, perfect sensitivity for all values of specificity):
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/roc-3-1.png" width = 400 height = 200>

* ROC curves are quite good for comparing methods (guessing or height cutoff) but neither of the measures plotted depend on prevelance. In these cases (where prevelance matters), we might make a *precision-recall plot*, its similar to ROC but precision is plotted against recall:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/precision-recall-1-1.png" width = 400 height = 200>
The reason guessing method is higher for males than females is because of the bias of more male data than female data.

Conditional Probabilities:
* In machine learning algorithms we can't predict outcomes perfectly all the time. The most common reason for this is because it's impossible. Most datasets will include groups of observations with the same exact observed values for all predictors, resulting in the same prediction. But, they have different outcomes making it impossible to make the predictions right for all these observations. For example, for any given height (x) you will have both males and females x inches tall so you can't predict them all right. But, we can still build algorithms much better than guessing and, in some cases, better than expert opinion.
* To achieve this in an optimal way, we make use of probabilistic representations of the problem. Observations with the same observed values for the predictors may not all be the same but, we can assume they all have the same probability of this class or that class. 
* The probabilistic representations can be written out mathematically for categorical data: (X<sub>1</sub> = x<sub>1</sub>, ..., X<sub>p</sub> = x<sub>p</sub>) for observed values <strong>x</strong> = (x<sub>1</sub>, ..., x<sub>p</sub>) and for covariates/predictors <strong>X</strong> = (X<sub>1</sub>, ..., X<sub>p</sub>). This doesn't imply the outcome (y) will take a specific value but, rather, it implies a specific probability. The *conditional probabilities* for each class (k) are mathematically denoted as: Pr(Y = k | X<sub>1</sub> = x<sub>1</sub>, ..., X<sub>p</sub> = x<sub>p</sub>), for k = 1, ..., K. Using the bolded letters the mathematical denotion can be rewritten as: pk(<stron>x</strong>) = Pr(Y = k | <strong>X</strong> = <strong>x</strong> ), for k = 1, ..., K. p(x) will represent conditional probabilities as functions, make sure to not confuse this with the p that's used to represent the number of predictors.
* Knowing these probabilities can help guide the construction of an algorithm that makes the best prediction. For any given <strong>X</strong> (set of predictors) predict the class k with the largest probability among p<sub>1</sub>(x), p<sub>2</sub>(x), ..., p<sub>K</sub>(x) with the mathematical notation being: Ŷ = max<sub>k</sub>p<sub>k</sub>(x) (<- Bayes Rule). But, we don't know the p<sub>k</sub>(x), in fact, this is one of the main problems of machine learning. 
* The better the algorithm estimates p̂<sub>k</sub>(x), the better the predictor, Ŷ = max<sub>k</sub>p̂<sub>k</sub>(x), will be. How good the prediction wil be will depend on 2 things, how close the maximum probability (max<sub>k</sub>p̂<sub>k</sub>(x)) is to 1 and how close the estimates, p̂<sub>k</sub>(x), are to the acutal probabilites, p<sub>k</sub>(x). Nothing can be done about the 1st restriction (determined by the nature of the problem) but, for the 2nd one we need to find the best way to estimate conditional probabilities. While some algorithms can get perfect accuracy (digit readers) other have success restricted by the randomness of the process (1st restriction). Also, defining the prediction by maximing the probability isn't always optimal and depends on the context, like sensitivity and specificity may differ in importance. But, even in these cases, having a good estimate of conditional proabilities will suffice to build an optimal prediction model since sensitivity and specificity can be controlled.
* Pr(Y = 1 | <strong>X</strong> = <strong>x</strong>) as the proportion of 1s in the stratum of the population for which <strong>X</strong> = <strong>x</strong>. Many algorithms can be applied to continous and categorical data due to the connection between conditional probabilities and conditional expectations. 
* The *conditional expectation* is the average of values (y<sub>1</sub>, ..., y<sub>n</sub>) in the population. In the case, which the y's are 0s or 1s the expectation is equivalent to the probability of randomly picking a 1 since the average is the proportion of 1s. Therefore, the conditional expectation is equal to the conditional probability, E(Y ∣ <strong>X</strong> = <strong>x</strong>) = Pr(Y = 1 ∣ <strong>X</strong> = <strong>x</strong>). Because of that, the conditional expectation is usually only used to denote both conditional expectation and probability.
* Just like with categorical outcomes, in most applications, the same observed predictors don't guarantee the same continuous outcome. Instead, we assume the outcome follows the same conditional distribution. For continuous outcomes, the best algorithm is based on a *loss function*. The most common one is *squared loss function*, Ŷ = predictor and Y = actual outcome, the squared loss function finds the square of the difference, (Ŷ - Y)<sup>2</sup>. Since there's usually a test set with many observations, n observations, the *mean squared error* is used. If the outcomes are binary, both RMSE and MSE are equivalent to 1 minus accuracy, since (y - y)<sup>2</sup> equals either 0 (correct prediction) or 1 (incorrect prediction).
* In general, the goal of an algorithm is to minimize the loss so it's close to 0 as possible. Since, the data is usually a random sample the MSE is a random variable. It's possible an algorithm could minimize the MSE on a particular dataset due to luck so we try to find an algorithm that minimizes the MSE *on average* (an algorithm that minimizes the average of the squared loss across many random samples). Note: this is a theoretical concept since we only have 1 dataset to work with so we can't have many random samples. But, there are techniques to estimate this quantitiy. 
* The reason conditional expectation is used in machine learning is because the expected value minimizes the MSE Ŷ = E (Y | <strong>X</strong> = <strong>x</strong>) minimizes E{(Ŷ − Y)<sup>2</sup> | <strong>X</strong> = <strong>x</strong>}. Due to this property the main task of machine learning can be described as: use data to estimate conditional probabilities, f(x) = E(Y | <strong>X</strong> = <strong>x</strong>) for any set of features <strong>x</strong> = (x<sub>1</sub>, ..., x<sub>p</sub>). The main way in which competing machine learning algorithms differ is in their approach to estimating this expectation.
