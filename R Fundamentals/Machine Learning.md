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
* *Specificity* is the ability of an algorithm to predict a negative when the observed outcome is negative (Ŷ = 0 when Y = 0). High specificity: Y = 0 -> Ŷ = 0. Another way to define specificity is the portion of positive calls that are actually positive. In this case, High specificity: Ŷ = 1 -> Y = 1. Also, specificity() can be used to find the specificity of a prediction: ```specificity(predicted_outcome_data, actual_data)```.
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
* Knowing these probabilities can help guide the construction of an algorithm that makes the best prediction. For any given <strong>X</strong> (set of predictors) predict the class k with the largest probability among p<sub>1</sub>(x), p<sub>2</sub>(x), ..., p<sub>K</sub>(x) with the mathematical notation being: Ŷ = max<sub>k</sub>p<sub>k</sub>(x). But, we don't know the p<sub>k</sub>(x), in fact, this is one of the main problems of machine learning. 
* The better the algorithm estimates p̂<sub>k</sub>(x), the better the predictor, Ŷ = max<sub>k</sub>p̂<sub>k</sub>(x), will be. How good the prediction wil be will depend on 2 things, how close the maximum probability (max<sub>k</sub>p̂<sub>k</sub>(x)) is to 1 and how close the estimates, p̂<sub>k</sub>(x), are to the acutal probabilites, p<sub>k</sub>(x). Nothing can be done about the 1st restriction (determined by the nature of the problem) but, for the 2nd one we need to find the best way to estimate conditional probabilities. While some algorithms can get perfect accuracy (digit readers) other have success restricted by the randomness of the process (1st restriction). Also, defining the prediction by maximizing the probability isn't always optimal and depends on the context, like sensitivity and specificity may differ in importance. But, even in these cases, having a good estimate of conditional proabilities will suffice to build an optimal prediction model since sensitivity and specificity can be controlled.
* Pr(Y = 1 | <strong>X</strong> = <strong>x</strong>) as the proportion of 1s in the stratum of the population for which <strong>X</strong> = <strong>x</strong>. Many algorithms can be applied to continous and categorical data due to the connection between conditional probabilities and conditional expectations. 
* The *conditional expectation* is the average of values (y<sub>1</sub>, ..., y<sub>n</sub>) in the population. In the case, which the y's are 0s or 1s the expectation is equivalent to the probability of randomly picking a 1 since the average is the proportion of 1s. Therefore, the conditional expectation is equal to the conditional probability, E(Y ∣ <strong>X</strong> = <strong>x</strong>) = Pr(Y = 1 ∣ <strong>X</strong> = <strong>x</strong>). Because of that, the conditional expectation is usually only used to denote both conditional expectation and probability.
* Just like with categorical outcomes, in most applications, the same observed predictors don't guarantee the same continuous outcome. Instead, we assume the outcome follows the same conditional distribution. For continuous outcomes, the best algorithm is based on a *loss function*. The most common one is *squared loss function*, Ŷ = predictor and Y = actual outcome, the squared loss function finds the square of the difference, (Ŷ - Y)<sup>2</sup>. Since there's usually a test set with many observations, n observations, the *mean squared error* is used. If the outcomes are binary, both RMSE (the square root of MSE) and MSE are equivalent to 1 minus accuracy, since (y - y)<sup>2</sup> equals either 0 (correct prediction) or 1 (incorrect prediction).
* In general, the goal of an algorithm is to minimize the loss so it's close to 0 as possible. Since, the data is usually a random sample the MSE is a random variable. It's possible an algorithm could minimize the MSE on a particular dataset due to luck so we try to find an algorithm that minimizes the MSE *on average* (an algorithm that minimizes the average of the squared loss across many random samples). Note: this is a theoretical concept since we only have 1 dataset to work with so we can't have many random samples. But, there are techniques to estimate this quantitiy. 
* The reason conditional expectation is used in machine learning is because the expected value minimizes the MSE Ŷ = E (Y | <strong>X</strong> = <strong>x</strong>) minimizes E{(Ŷ − Y)<sup>2</sup> | <strong>X</strong> = <strong>x</strong>}. Due to this property the main task of machine learning can be described as: use data to estimate conditional probabilities, f(x) = E(Y | <strong>X</strong> = <strong>x</strong>) for any set of features <strong>x</strong> = (x<sub>1</sub>, ..., x<sub>p</sub>). The main way in which competing machine learning algorithms differ is in their approach to estimating this expectation.

Example code of conditional proabilities:
```r
set.seed(1)
disease <- sample(c(0,1), size=1e6, replace=TRUE, prob=c(0.98,0.02)) # The prevelance of the disease is 0.02
test <- rep(NA, 1e6)
test[disease==0] <- sample(c(0,1), size=sum(disease==0), replace=TRUE, prob=c(0.90,0.10)) # The test is negative 90% of the time when tested on a healthy paitent.
test[disease==1] <- sample(c(0,1), size=sum(disease==1), replace=TRUE, prob=c(0.15, 0.85)) # The test is positive 85% of the time when tested on a patient with the disease.
# Probabiliy test is positive (0.11):
mean(test == 1)
# Probabiliy indivual has disease if test is negative (0.03):
mean(disease[test == 0])
# Probability indivual has disease if test is positive (0.14):
mean(disease[test == 1])
# If a patient tests positive how much does that increase the chance of having the disease (~7):
mean(disease[test==1]==1)/mean(disease==1)
```
<strong>Linear Regression for Prediction, Smoothing, and Working with Matricies:</strong>

Linear Regression for Prediction:
* Linear regression can be considered a machine learning algorithm since it's too rigid to be usefule in general but can work quite well for some challenges. It, also, serves as a baseline approach, if you can't beat it with a more complex approach, you probably want to stick to linear regression. 
* Your tasked with building a machine learning algorithm that predicts son's height (Y) given the father's height (X), from the Galton heights dataset. We can generate some testing and training sets. We already created a linear regression model to predict the son's height based on the father's. So we can use that fitted model as a machine learning algorithm, this gives us the conditional expectation, f(x) = 38 + 0.47x, and the squared loss is 4.78:
```r
# Load librarires and filter and name columns for galton_heights:
library(HistData)
set.seed(1983)
galton_heights <- GaltonFamilies %>%
  filter(gender == "male") %>%
  group_by(family) %>%
  sample_n(1) %>%
  ungroup() %>%
  select(father, childHeight) %>%
  rename(son = childHeight)
# Get the training and test sets and set y = galton_height$son, since we're solving for y (son):
y <- galton_heights$son
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
train_set <- galton_heights %>% slice(-test_index)
test_set <- galton_heights %>% slice(test_index)
# fit linear regression model
fit <- lm(son ~ father, data = train_set)
fit$coef
y_hat <- fit$coef[1] + fit$coef[2]*test_set$father
# Find the squared loss (4.78):
mean((y_hat - test_set$son)^2)
```
* The predict() function is very useful for machine learning algorithms. The function takes fitted objects (such as lm() or glm()) and a data frame with the new predictors for which to predict, returning the prediction. So, instead, of writing out the formula for the regression line, the predict() function can be used: ```y_hat <- predict(fit, test_set)```.
* The regression approach can be applies to categorical data. We can use the previous example of predicting sex based on heights to illustrate this.
* The outcome Y = 1 for females and Y = 0 for males with X = height. We're interested in the conditional probability, Pr(Y = 1 | X = x), being female given the height. What's the conditional probability of being female if your 66 inches tall, Pr(Y = 1 | X = 65). You can use this code:
```r
library(dslabs)
data("heights")
y <- heights$height

set.seed(2) #if you are using R 3.5 or earlier
set.seed(2, sample.kind = "Rounding") #if you are using R 3.6 or later
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
train_set <- heights %>% slice(-test_index)
test_set <- heights %>% slice(test_index)

train_set %>% 
  filter(round(height)==66) %>%
  summarize(y_hat = mean(sex=="Female"))
```
* And, the conditional probability is ~ 24%. We do this for several values, plotting them, and they look close to linear we'll try to use regression. We can use this conditional probability: p(x) = Pr(Y = 1 | X = x) = β<sub>0</sub> + β<sub>1</sub>x (intercept + slope * height). Converting the factors (sex) to 0s and 1s we can estimate β<sub>0</sub> and β<sub>1</sub> using least squares: ```lm_fit <- mutate(train_set, y = as.numeric(sex == "Female")) %>% lm(y ~ height, data = .)```. To form a prediction we create a *decision rule*, we predict female if the conditional probability > 0.5: 
```r
p_hat <- predict(lm_fit, test_set) 
y_hat <- ifelse(p_hat > 0.5, "Female", "Male") %>% factor()
confusionMatrix(y_hat, test_set$sex)$overall["Accuracy"]
# Using the confusion matrix we see we got an accuracy rate of 78.5%.
```
* The estimate we obtained for our conditional probability using linear regression goes from -0.4 to -1.2. *Logistic Regression* is an extension of linear regression that assures the estimate of the conditional probability is between 0 and 1, making using of the logistic transformation (g(p) = log(p/1 - p)). The logistic transformation converts proabilities into log odds, the odds tell us how much more likely something will happen compared to not happening. For example, if p = 0.5, the odds are 1 to 1, if p = 0.75, the odds are 3 to 1.
* How do we fit this new model, we can't use least squares anymore. Instead we compute the *maximum likelyhood estimate* using the glm(), stands for general linearized models and fits the logistic transformation model: 
```r
glm_fit <- train_set %>% 
  mutate(y = as.numeric(sex == "Female")) %>%
  glm(y ~ height, data=., family = "binomial")
# Specify the model using the family parameter.
```
* We can, also, use the predict() function on the logistic regression model but we need to specify the type parameter to "response" so we get the conditional probabilities back:```p_hat_logit <- predict(glm_fit, newdata = test_set, type = "response")```. Looking at the confusion matrix we see our accuracy has been raised to 80%.
* Both, linear and logistic regression provide an estimate for the conditional expectation, which, in the case of binary data, is the conditional probability. 
* The algorithms we've made up above aren't technically categorized as machine learning algorithms since our algorithms have only 1 predictor while true machine learning algorithms have many. So, we can go back to the digits example which has 784 predictors but, we're only going to look at a subset of this data which has only 2 predictors and categories. 
* We need to build an algorithm that can detect if a digit is a 2 or a 7 from the predictors. These predictors will be the proportion of dark pixels in the upper left quadrant and the proportion of pixels in the lower right quadrant. Also, to have a more manageable dataset we'll select 1,000 digits (500 in training set and 500 in test set) from the training set which has 60,000 digits.
* Plotting the predictors with the outcomes against each other reveals some information. If x_1 (the 1st predictor, upper left quadrant) is large the digit is probably a 7. Also, in x_2 (the 2nd predictor, the lower right quadrant) the 2's appear to be mid range values. 
* We can see the digits with the largest and smallest x_1, the largest one is a 7 and the smallest one is a 2. Now, we see the digits with the largest and smallest x_2, they're both 7s. So we start getting a sense for why these predictors are informative and why they're challenging. 
* For the machine learning algorithm we can just use logistic regression, the conditional probability of being a 7 given the 2 predictors, x<sub>1</sub> and x<sub>2</sub>: p(x<sub>1</sub>,x<sub>2</sub>) = Pr(Y = 1 | X<sub>1</sub> = x<sub>1</sub>, X<sub>2</sub> = x<sub>2</sub>) = g<sup>-1</sup>(β<sub>0</sub> + β<sub>1</sub>x<sub>1</sub> + β<sub>2</sub>x<sub>2</sub>) with g<sup>-1</sup> the inverse of the logisitic function, g<sup>-1</sup>(x) = exp(x)/{1 + exp(x)}.
* This can be fitted by using glm(): ```fit_glm <- glm(y ~ x_1 + x_2, data=mnist_27$train, family = "binomial")```. Now, we can build a decision rule based on the conditional probability, whenever the probability is bigger than 0.5, p > 0.5, we predict a 7 and if it's not we predict a 2:
```r
p_hat_glm <- predict(fit_glm, mnist_27$test)
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5, 7, 2))
confusionMatrix(data = y_hat_glm, reference = mnist_27$test$y)$overall["Accuracy"]
```
* By analyzing the confusion matrix we see we achieve an accuracy of 79%. We can access and plot the true conditonal probability, since a person already found the true conditional probability and put it in the datset for us (graph on the right):
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/best-knn-fit-1.png" width = 600 height = 400>

* We can see the true conditional probability up above. But, we can compare the true conditional probability to the estimated conditional probability. We can find the boundary where the conditonal probability is 0.5 (left graph) using logistic regression:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/lda-estimate-1.png" width = 400 height = 200>

* The boundary can't be anything other than a straight line which means logistic regression can't capture the non-linear nature of the true conditional probability. We can see where the mistakes were made, by plotting, and see that because the 2s and 7s are divided by a line, logistic regression misses several values that cannot be captured by just a line. 
* So we need something more flexible since logistic regression forces the boundaries to be a line or something straight. We need something that can permit other shapes. This can be accomplished with the *nearest neighbor algorithm* and some *kernal* approaches, falling under smoothing.

Smoothing:
* *Smoothing* is a very powerful technique used across data analysis, other names are curve fitting and low pass filtering, that's desinged to detect trends in the presence of nosiy data in cases of which the shape of the data is unknown. To accomplish this we assume the trend is smooth and the noise is unpredictably wobbly:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/signal-plus-noise-example-1.png" width = 400 height = 300>

* Assumptions help us extract a trend from the noise. The concepts of smoothing are extremley useful in machine learning because the conditional expectations and contional probabilities can be thought of as trends of unknown shapes that we need to estimate in the presence of uncertainty.
* We're going to try to estimate the time trend in the popular vote from the 2008 election, the difference between Obama and McCain. We can load the data and plot it (we're trying to learn the shape of the trend, after collecting the data):
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/polls-2008-data-1.png" width = 300 height = 200>

* We assume for any given day (x) there's a true preference among the electorate (f(x)) but due to the uncertainty of polling each data point comes with an error (ε), a mathematical model beign: Y<sub>i</sub> = f(x<sub>i</sub>) + ε<sub>i</sub>. We want to predict Y given the day x and, if we knew it, we would use the conditional expectation, f(x) = E(Y | X = x), but we don't know it, so we have to estimate it. 
* We can start by using regression:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/linear-regression-not-flexible-1.png" width = 400 height = 200>

* The regression line does't seem to fit the trend very well. For example, on day -62 the Republican Convention was held and that appeared to give McCain a boost in the election, mimicked by the data, but the regression line doesn't show this. Thus, we need an alternative, more flexible approach. 
* *Bin smoothing* is to group data points into strata in which the value of f(x) can be assumed to be constant. We can make this assumption because we think f(x) changes slowly and f(x) is almost constant in small windows of time.
* An example of this is to assume, for the poll data, that public opinion remains approximatley the same within a week's time. With this, we have several data points with the same expected value. So we can fix a day to be the center of the week, x<sub>0</sub>, and for any other day x that |x - x<sub>0</sub>| ≤ 3.5 we assume f(x) is a constant, f(x) = μ. This assumption implies, the expected value of Y given x is approximatley μ when the distance between x<sub>i</sub> and x<sub>0</sub> is ≤ 3.5, mathematically: E[Y<sub>i</sub> | X<sub>i</sub> = x<sub>i</sub>] = μ if |x<sub>i</sub> - x<sub>0</sub>| ≤ 3.5. In smoothing, the size of the interval satisfying the condition of the distance between x<sub>i</sub> and x<sub>0</sub> is ≤ 3.5, is referred to as the *window size, the bandwith, or the span*. This assumption implies a good estimate of f(x) is the average of the y values in the window. We can define A<sub>0</sub> as the set of indexes i such that |x<sub>i</sub> - x<sub>0</sub>| ≤ 3.5 and N<sub>0</sub> as the number of indexes in A<sub>0</sub> then the estimate is: f(x<sub>0</sub>) = 1/N<sub>0</sub> * ∑<sub>i∈A<sub>0</sub></sub>Y<sub>i</sub> (average in the window). The idea behind bin smoothing is to make this calculation for each value of x, so we make each value of x the center and then recompute the average. 
* So, in the poll example we would compute the average for the values within a week of the day we're considering. We can set the center at -125 and -55 (black points are the points used to compute the averages at their respective points, blue line represents the computed average):
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/binsmoother-expained-1.png" width = 400 height = 300>

* By computing the average for each point we can form an estimate for the underlying curve f(x):
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/binsmoother-final-1.png" width = 400 height = 300>

Code:
```r
# bin smoothers
span <- 7 
fit <- with(polls_2008,ksmooth(day, margin, x.points = day, kernel="box", bandwidth =span))
polls_2008 %>% mutate(smooth = fit$y) %>%
    ggplot(aes(day, margin)) +
    geom_point(size = 3, alpha = .5, color = "grey") + 
    geom_line(aes(day, smooth), color="red")
```
* The final result for the bin smoother is quite wiggly because each time the window moves the 2 points change (and we start with 7 points, meaning a substanial change). We can reduce the wiggle a little by taking weighted averages that give the center of a point more weight than those that are far away from the center, with the points at the edges recieving very little weight. We call the functions from which we compute these weights the *kernel* (the bin smoother can be thought of as an approach that uses the kernel). Formula: f(x<sub>0</sub>) = <sub>i = 1</sub>∑<sup>N</sup>w<sub>0</sub>(x<sub>i</sub>)Y<sub>i</sub>. Each point recieves a weight, in the case of bin smoothers, between 0 for points that are outside the window and 1/N<sub>0</sub> for points inside the window, with N<sub>0</sub> the number of points in that week. For the graph, we used the ```kernel = "box"``` in the function ksmooth() to get it a little boxy. But, we can use ```kernel = normal``` to attain a smoother estimate by using the normal/Gaussian density to assign weights:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/gaussian-kernel-1.png" width = 400 height = 300>

* Now, we can apply this to the final estimate for the underyling curve f(x) using normal/Gaussian density to assign weights:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/final-ksmooth-normal-kernel-1.png" width = 400 height = 300>

Code:
```r
# kernel
span <- 7
fit <- with(polls_2008, ksmooth(day, margin,  x.points = day, kernel="normal", bandwidth = span))
polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = .5, color = "grey") + 
  geom_line(aes(day, smooth), color="red")
```
* There are several functions in R that implement bin smoothers or kernel approaches (ksmooth()). In practice, we typically prefer slightly more complex models than fitting a constant. The smooth binsmoother graph, up above, with weighted points is still kinda wiggly. We're going to learn how to improve them. 
* A limitation of the bin smoother is that we need small windows for the approximatley constant assumption to hold. As a result, we end up with a small number of data points to average and because of this we obtain imprecise estimates of the trend. 
* *Local weighted regression or loess* permits us to conisder larger windows. To do this we'll use the mathematical result, know as Taylor's theroem, which tells us if you look close enough at any smooth function (f) it looks like a line. So, instead of assuming the function is constant in a window we assume it's locally linear. With the linear assumption we can consider larger window sizes than with the constant. 
* We start with a 3 week window and the model for points in this window are denoted as (we assume Y given X in that window is a line): E[Y<sub>i</sub> | X<sub>i</sub> = x<sub>i</sub>] = β<sub>0</sub> + β<sub>1</sub> * (x<sub>i</sub> - x<sub>0</sub>) if |x<sub>i</sub> - x<sub>0</sub>| ≤ 10.5. Now, for every point (x<sub>0</sub>) loess defines a window and then fits a line within that window. The loess on 2 points, -125 and -55:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/loess-1.png" width = 400 height = 300>

* The fitted values at x<sub>0</sub> become the trend, it's smoother than the bin and kernel fit because we used larger sample sizes:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/final-loess-1.png" width = 400 height = 300>

* Different spans/window sizes give different estimates:
<img src = "https://rafalab.github.io/dsbook/ml/img/loess-multi-span-animation.gif" width = 500 height = 400>

<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/loess-final-1.png" width = 500 height = 400>

* With 0.1 it's quite wiggly, 0.15 is slightly better, 0.25 is really smooth, and 0.66 almost looks like a straight line. 
* There are 3 more differences between a loess and a bin smoother:
	1. Rather than keeping the bin size the same, loess keeps the number of points used in the local fit the same. This number is controlled via the span argument which excpects a proportion. For example, if N is a number of data points and the span is 0.5 then for any given X, loess will use 0.5 * N closest points for the fit.
	2. When fitting a line locally, loess uses a weighted approach. Instead of least squares loess minimizes a weighted version. Instead of the Gaussian kernel, loess uses a function called the Turkey tri-weight: W(u) = (1 - |u|<sup>3</sup>)<sup>3</sup> if |u| ≤ 1 and W(u) = 0 if |u| ≤ 1, weights defined as: w<sub>0</sub>(x<sub>i</sub>) = W(x<sub>i</sub> - x<sub>0</sub> / h). The kernel for the tri-weight looking like this:
	<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/triweight-kernel-1.png" width = 400 height = 300>
	
	3. The loess has the otpion of fitting the local model robustly. An iterative algorithm is used in which, after fitting a model in 1 iteration, outliers are detected and down-weighted for the next iteration. To use this option use the argument ```family = "symmetric"````
	4. One extra point about loess. Taylor's theroem tells us that if you look at a function close enoug, it looks like a parabola and you don't have to look as close for the linear approximation. This means we can make our windows even larger and fit parabolas instead of lines, so the local model would be: E[Y<sub>i</sub> | X<sub>i</sub> = x<sub>i</sub>] = β<sub>0</sub> + β<sub>1</sub>(x<sub>i</sub> - x<sub>0</sub>) + β<sub>2</sub>(x<sub>i</sub> - x<sub>0</sub>)<sup>2</sup> if |x<sub>i</sub> - x<sub>0</sub>| ≤ h (this is the default procedure for loess). The paramater degree, tells loess what polynomials of degree to fit, like ```degree = 1``` would fit a polynomial of degree 1, a straight line. But, the degree value defaults to 2 so if nothing is put loess will fit parabolas (polynomials of degree 2). Comparision of fitting lines (red dashed) and fitting parabolas (orange solid):
	<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/polls-2008-parabola-line-loess-1.png" width = 400 height = 300>
	
	* Degree = 2 gives a more wiggly result and degree = 1 is less prone to the noise that affects parabolas. 
* ggplot uses loess in the goem_smooth() function. So the code below will result in a fitted graph:
```r
polls_2008 %>% ggplot(aes(day, margin)) +
  geom_point() + 
  geom_smooth(color="red", span = 0.15, method.args = list(degree=1))
```
Graph:

<img src = "https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/loes_fitted.PNG" width = 400 height = 300>

Example of Using loess to plot:
```r
# Wranlge the Data:
library(tidyverse)
library(lubridate)
library(purrr)
library(pdftools)
    
fn <- system.file("extdata", "RD-Mortality-Report_2015-18-180531.pdf", package="dslabs")
dat <- map_df(str_split(pdf_text(fn), "\n"), function(s){
	s <- str_trim(s)
	header_index <- str_which(s, "2015")[1]
	tmp <- str_split(s[header_index], "\\s+", simplify = TRUE)
	month <- tmp[1]
	header <- tmp[-1]
	tail_index  <- str_which(s, "Total")
	n <- str_count(s, "\\d+")
	out <- c(1:header_index, which(n==1), which(n>=28), tail_index:length(s))
	s[-out] %>%
		str_remove_all("[^\\d\\s]") %>%
		str_trim() %>%
		str_split_fixed("\\s+", n = 6) %>%
		.[,1:5] %>%
		as_data_frame() %>% 
		setNames(c("day", header)) %>%
		mutate(month = month,
			day = as.numeric(day)) %>%
		gather(year, deaths, -c(day, month)) %>%
		mutate(deaths = as.numeric(deaths))
}) %>%
	mutate(month = recode(month, "JAN" = 1, "FEB" = 2, "MAR" = 3, "APR" = 4, "MAY" = 5, "JUN" = 6, 
                          "JUL" = 7, "AGO" = 8, "SEP" = 9, "OCT" = 10, "NOV" = 11, "DEC" = 12)) %>%
	mutate(date = make_date(year, month, day)) %>%
        filter(date <= "2018-05-01")
# Using loess() to obtain a smooth estimate of the expected number of deaths as a function of date:
span <- 60 / as.numeric(diff(range(dat$date)))
fit <- dat %>% mutate(x = as.numeric(date)) %>% loess(deaths ~ x, data = ., span = span, degree = 1)
dat %>% mutate(smooth = predict(fit, as.numeric(date))) %>%
	ggplot() +
	geom_point(aes(date, deaths)) +
	geom_line(aes(date, smooth), lwd = 2, col = "red")
# Using loess() to plot estimates against day of year with different colors for the year:
dat %>% 
    mutate(smooth = predict(fit, as.numeric(date)), day = yday(date), year = as.character(year(date))) %>%
    ggplot(aes(day, smooth, col = year)) +
    geom_line(lwd = 2)
```
Working With Matricies:
* In machine learning, where all predictors are numeric or can be converted into numerics in a meaningful way, are common. For example, in the digits dataset every pixel records a number between 0 and 255.
* The 60,000 digits can be loaded and in these cases (because of the sheer size) it's convient to save the predictors in a matrix and the outcomes in a vector rather than using a data frame. In fact, the training data for the digits example does this since the class of mnist$train$images (```class(mnist$train$images)```) is a matrix. 
* We can take the first 1,000 digits for the training and testing sets, to make the data sets more manageable. ```x <- mnist$train$images[1:1000,] y <- mnist$train$labels[1:1000]```.
* In machine learning, the main reason for using matricies is that certain mathematical operations needed to develop efficent code can be performed using techniques from linear algebra.
* In linear algebra, there are scalars (a number, a = 1), vectors (one column matricies containing a lot of salar entries), and matricies (kind of like a box cotaining a lot of data). The dimensions of a matrix (rows by columns, think RC Cola) can be extracted in R by using the dim(): ```dim(matrix)``` and a vector in R doesn't have any dimensions but, the vector can be converted into a matrix using the as.matrix(): ```as.matrix(vector)``` and now it has dimensions. 
* It's often useful to convert vectors into matricies. For example, since the variables are pixels on a grid we can convert the rows of pixel intesities into a matrix representing the grid. We can use the matrix() function for this and specifying the number of rows and columns the resulting matrix should have (the matrix is filled by columns, 1st, 2nd, and so on):```my_vector <- 1:15 mat <- matrix(my_vector, 5, 3) #Creates a matrix named mat from my_vector with 5 rows and 3 colummns```. We can fill in the matrix with the byrow argument to fill it by row:```mat_t <- matrix(my_vector, 3, 5, byrow = TRUE) #This creates a matrix of 3 rows and 5 columns, filled in by rows```. The function t() can be used to directly transpose (flip columns and rows) a matrix.
* Note: the matrix() function recycles values in the vector WITHOUT warning if the product of columns and rows does not match the length of the vector. But, we can use this for our advantage in practice. For example, we want to put the pixel intensitites of the 3rd entry (represents the digit, 4) into a grid, we can use this: ```grid <- matrix(x[3,], 28, 28)```. To confirm we've done this correctly, we can use the function, image(), which shows an image of the 3rd agrument. Also, we'll flip the image because R plots the image starting from pixel one, so code will be: ```image(1:28, 1:28, grid[, 28:1])```.
* For the total pixel darkness we want to sum the values of each rown and then visualize how these values vary by digits. The function rowSums() takes a matrix as input and takes the sum of each row: ```sums <- rowSums(x)```. Also, we can compute the average of each row using rowMeans(): ```avg <- rowMeans(x)```. With this we can generate a box plot to see how the average pixel intesity changes from digit to digit:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/boxplot-of-digit-averages-1.png" width = 500 height = 300>

```r
data_frame(labels = as.factor(y), row_averages = avg) %>%
    qplot(labels, row_averages, data = ., geom = "boxplot")
```
* We can, also, compute the columns sums and means using colSums() and colMeans(), respectivley. The package, matrixStats, has a lot of useful functions to perform on rows and columns, efficently. There are functions to find the standard deviation estimates on the rows and columns, rowSds() and columnSds(), there are a lot more useful functions. The apply() function lets you apply any function to a matrix: ```appply(matrix, dimension, function) #In the dimension arugment, 1 = rows and 2 = columns```. But, these apply() functions aren't as fast as the dedicated functions.
* Now, lets study the variation of each pixel and remove the columns with pixels that don't change much. We'll quantify the variation of each pixel using the standard deviation across all entries. Since each column represent a pixel we can use the colSds() function: ```sds <- colSds(x)```. We can look at the distribution of the pixels:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/sds-histogram-1.png" width = 400 height = 300>

* Looking at the plot you can see some pixels have very low entry to entry variability. this make sense since we don't write in some parts of the box. Variance plotted by location:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/pixel-variance-1.png" with = 400 height = 300>

* There is little variation in the corners because the digits are written in the center. We can remove pixels that don't have any variation because they're useless to us. We can remove them using this: ```new_x <- x[ ,colSds(x) > 60]```. Important note: when subsetting matricies, selecting 1 row or 1 column will result in a vector but, the martix can be preserved by using the argument drop: ```class(x[ , 1, drop=FALSE]) #class is martix```.
* Now, we want to look at a histogram of all the pixels. We can turn marticies into vectors using the function as.vector(): ```as.vector(matrix)```. Now, we can see a histogram of all the predictors:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/histogram-all-pixels-1.png" with = 400 height = 300>

* We can see a clear dichotomy as parts with ink and without ink. If we think values less than 25 are smudges we can make them 0: ```new_x[new_x < 25] <- 0```. The histogram suggest pixels are either ink or no ink (binary) so we can binarize the data. We can binarize the data using matrix operations (we don't lose too much information):
```r
#binarize the data
bin_x <- x
# Values below 255/2 turn to 0 and those above or equal to turn to 1. 
bin_x[bin_x < 255/2] <- 0
bin_x[bin_x >= 255/2] <- 1
# Or it can be converted into a matrix using logicals and coerced into numbers:
bin_X <- (x > 255/2)*1
```
* Now, since we're standarizing the rows or columns we're going to use *vectorization*. If we subtract a vector from a matrix, the first element of each vector is subtracted form first row of the matrix, and so on. The same holds true for other arithmetic operations, so we can scale each row of a matrix like this: ```(x - rowMeans(x)) / rowSds(x)```. This doesn't work for columns and we would have to transpose a matrix, like this: ```t(t(x) - colMeans(x))```. Another function, sweep() takes each entry of a vector from the corresponding row or column: ```x_mean_0 <- sweep(x, 2, colMeans(x)) #Subtracts column mean from each column```. Even though, the default function for sweep() is subtraction you can change it using the FUN argument: ```x_standardized <- sweep(x_mean_0, 2, colSds(x), FUN = "/") # Divide by the standard deviation```.
* Matrix mulitplication is done with %*%, like this: ```t(x) %*% x```. Or we can compute the cross product of matrix with the crossprod() function: ```crossprod(x)```. To find the inverse of a function we can use solve(): ```solve(crosspod(x))```. And, a qr decomposition can be accessed as: ```qr(x)```.

<strong>Distance, Knn, Cross Validation, and Generative Models:</strong>

Nearest Neighbors:
* When we cluster animals into subgroups (reptiles, amhphibians, and mammals) we're implicilty defining a distance that allows us to say what animals are close to each other. Many machine learning techniques rely on being able to define distance between observations using features or predictors. 
* We're going to look at a random sample of 2s and 7s from mnist:
```r
if(!exists("mnist")) mnist <- read_mnist()
ind <- which(mnist$train$labels %in% c(2,7)) %>% sample(500)
#the predictors are in x and the labels in y
x <- mnist$train$images[ind,]
y <- mnist$train$labels[ind]
```
* For examples, like smoothing, we're interested in describing distances between observations (digits). Now, to find distance we need to know what the points are. With high dimensional data, points aren't on the Cartesian plan anymore, so we can't visualize them anymore and need to think abstractly. In our digits example, the predictors (<strong>X</strong><sub>i</sub>) are defined as points in 784 dimensional space: <strong>X</strong><sub>i</sub> = (x<sub>i, 1</sub>, ..., x<sub>i, 784</sub>). And, the Euclidian distance for distance between points 1 and 2 is given by this: dist(1, 2) = sqrt(<sup>784</sup>∑<sub>j = 1</sub> (x<sub>1, j</sub> - x<sub>2, j</sub>)<sup>2</sup>).
* Looking at the first 3 observations, ```y[1:3]``` we can see they're a 7, 7, and a 2. The vector for these 3 observations can be saved in 3 seperate objects:
```r
x_1 <- x[1,]
x_2 <- x[2,]
x_3 <- x[3,]
```
* We excepect the distances between the same number (7 and 7) to be smaller than the distances between 2 different numbers:
```r
sqrt(sum((x_1 - x_2)^2))
# Returns a distance of 2080
sqrt(sum((x_1 - x_3)^2))
# Returns a distance of 2252
sqrt(sum((x_2 - x_3)^2))
# Returns a distance of 2643
# A faster way to compute this is using crossprod() and matrix algebra:
sqrt(crossprod(x_1 - x_2))
sqrt(crossprod(x_1 - x_3))
sqrt(crossprod(x_2 - x_3))
# Also, we can compute all the distances between all the observations at once by using the function, dist().
# If you feed it a matrix, the dist() computes the distance between each row and produces an object of class dist:
d <- dist(x)
# Now, we can corece the dist object into a matrix:
as.matrix(d)[1:3,1:3]
# We can see an image of the distances:
image(as.matrix(d))
# Order the distance by labels:
image(as.matrix(d)[order(y), order(y)])
# Compute distance between predictors:
d <- dist(t(x))
dim(as.matrix(d))
# From the dim() the matrix is 784 by 784
d_492 <- as.matrix(d)[492,]
image(1:28, 1:28, matrix(d_492, 28, 28))
```
* The *k-nearest neighbor* estimates the conditional probabilities in a similar way to bin smoothing. However, kNN is easier to adapt to multiple dimensions. For any point which you want to estimate the conditional probability you look at the k-nearest points then find the average of these points, referred to as the *neighborhood*, due to the connection between condtional expectations and conditional probabilites, this gives us the estimated conditional probability. We can control the flexibility of the estimate via k, a large k results in a smoother estimate and a smaller k results in a more flexible but wigglier estimate.
* We'll go back to the digits dataset with 2s and 7s. We can think of the conditional probability of being a 7 (y<sub>1</sub>) given the 2 predictors: p(x<sub>1</sub>, x<sub>2</sub>) = Pr(Y = 1 | X<sub>1</sub> = x<sub>1</sub>, X<sub>2</sub> = x<sub>2</sub>). The 0s and 1s we observe are *noisy* because some of the regions of the conditional probability aren't close to 0 or 1, which means you can go either way sometimes. So we have to estimate the conditional probability, we can try to do k nearest neighbors for this. It'll be compared to logisitc regression, so to be better, it needs to beat out logistic regression. We can use to below code to find the accuracy for logistic regression:
```r
library(caret)
fit_glm <- glm(y~x_1+x_2, data=mnist_27$train, family="binomial")
p_hat_logistic <- predict(fit_glm, mnist_27$test)
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.5, 7, 2))
confusionMatrix(data = y_hat_logistic, reference = mnist_27$test$y)$overall[1]
```
* The accuracy for logistic regression is 76%. To use the knn algorithm we'll use the function knn3(), we can call it in 1 of 2 ways, the first specifying a formula (outcome ~ predictor_1 + predictor_2 + predictor_3, to use all the predictors do: y ~ .) and the data frame (all the data to be used): ```knn_fit <- knn3(y ~ ., data = mnist_27$train)```, the second way would be a matrix of predictors and a vector of the outcomes:
```r
x <- as.matrix(mnist_27$train[,2:3])
y <- mnist_27$train$y
knn_fit <- knn3(x, y)
```
* The 1st approach is a quicker, simpler way to write it when we're in a hurry. But, the 2nd appraoch is for large data sets. We need to pick the number of neighbors to include (the default is k = 5), we can use the default: ```knn_fit <- knn3(y ~ ., data = mnist_27$train, k=5)```. Since the data set is balanced (# of 2s and 7s are equal) and we care about specificity just as much as we care about specificity (both mistakes are equally bad) we'll use accuracy to quantify performance. The predict function produces either a probability for each class or it can produce the outcome with the highest probability: 
```r
y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class") # The fitted object is knn_fit and the type argument sets the predict function to return the probability for each class or just highest probability.
confusionMatrix(data = y_hat_knn, reference = mnist_27$test$y)$overall["Accuracy"]
```
* The confusion matrix returns an accuracy of 81.5% which is better than the logistic regression. 
* We can analyze why the knn did better than the logistic regression. Visualization of true conditional probability and knn with 5 neighbors estimate:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/knn-fit-1.png" width = 500 height = 300>

* The knn 5 estimate has the essence of the shape of the true probability unlike, the logistic regression. But, we can still do better since you can spot some blue specks in the red area which is due to *overtraining*. If you find the accuracies for training and test set you see that the accuracy for the training set is 88.2% while the test set is 81.5%, which is a pretty susbtantial difference. The reason for the big difference is because we overtrained. Overtraining is at its worst when k = 1, the estimate is obtained with just the y corresponding to that point (since only 1 point is the closest neighbor). When k = 1, we'll obtain practically perfect accuracy in the training set because each point is used to predict itself. Perfect accuracy will occur when there are enough unique predictors, the accuracy for the training data when k = 1 is 99.4% but the accuracy for the test set is 73.5%, lower than with logistic regression. The estimated conditional probability follows the training data too closely, messing up the accuracy for the test set. We see this overtaining with k = 5 so we should change it a larger k. Let's try k = 401 and the accuracy for the test data is 79%, close to logisitic regression:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/mnist-27-glm-est-1.png" width = 500 height = 300>

* The size of k is so large it doesn't permit enough flexibility for knn, almost half the data is used to compute each conditional probability. This is called *over smoothing*. To find the perfect value for k we can repeat and find the accuracies for many different values of k, we can use the map_df() to repeat for each k:
```r
ks <- seq(3, 251, 2)
library(purrr)
accuracy <- map_df(ks, function(k){
    fit <- knn3(y ~ ., data = mnist_27$train, k = k)
    y_hat <- predict(fit, mnist_27$train, type = "class")
    cm_train <- confusionMatrix(data = y_hat, reference = mnist_27$train$y)
    train_error <- cm_train$overall["Accuracy"]
    y_hat <- predict(fit, mnist_27$test, type = "class")
    cm_test <- confusionMatrix(data = y_hat, reference = mnist_27$test$y)
    test_error <- cm_test$overall["Accuracy"]
    
tibble(train = train_error, test = test_error)
    })
})
# We can graph it:
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/accuracy-vs-k-knn-1.png" width = 400 height = 300>

* The jaggedness is due to the fact the accuracy is computed on this sample and is a random variable. This is why it's preferred to minimize the expectation loss rather than the loss experienced with 1 data set. We can still draw out a pattern, low values of k give a low test set accuracy and high train set accuracy (overtraining), and large values of k result in low accuracy (over smoothing). We can find the maximum accuracy and optimal k:
```r
#pick the k that maximizes accuracy using the estimates built on the test data
ks[which.max(accuracy$test)]
max(accuracy$test)
```
* The maximum accuracy is 85% and the optimal k is 41. But, there is a big problem since we broke a very important rule of machine learning, we selected the k based on the test set. For this, we'll use cross-validation which provides a way to estimate the expected loss for a given method using only the training set.

Cross-Validation:
* With 1 dataset the MSE (mean squared error) can be estimated from the observed mean square error, referred to as the true error and apparent error, respectivley. There are 2 important characteristics of the apparent error, the dataset is a random variable so the apparent error is a random variable (so an algorithm having lower apparent error might be due to luck). Also, if we train an algorithm on the same dataset we used to compute the apparent error we might be overtraining, this will make the apparent error an underestimate of the true error. 
* Cross validation allows us to try to get rid of these problems. It helps to think of the true errror as the average of many apparent errors obtained by applying the algorithm to new, random samples of the data (none of them used to train the algorithm). The idea of cross validation is to imitate this theoretical setup the best we can with the data at hand. To do this we need to generate a series of random samples, the general idea for this is to randomly generate smaller data sets that aren't used for training but used to estimate the true error.
* One approach for cross validation is the *k-fold cross validation*. We divide our whole data set into training and testing sets, the training set will be used, exclusivley, for training the algorithm and the test set will be used to only evaluate the algorithm. The test set is usually a smaller portion of the data set so the training algorithm can be trained on as much data as possible. But, we need the test to be large so stable estimates of the loss can be obtained (typical size for test set is 10% - 20%). 
* Do NOT use the test set when training, not for filtering rows, not for selecting features, nothing. But, for most machine learning algorithms we need to select parameters, like the number of neighbors (k) in the k nearest neighbors algorithm. The set of parameters will be represented by λ. So, lambda (algorithm parameters) needs to be optimized without using the test or training (results in overtraining) tests. For each set of algorithm parameters being considered, we want an estimate of the MSE and then we will choose the parameters with the smallest MSE. It's important that before starting the cross-validation process the algorithm parameters are fixed. In k-fold cross validation, we randomly split the observations into k non-overlapping sets, and repeat the calculation for MSE for each of these sets. Then, we compute the average MSE and obtain an estimate of our loss. Finally, we can select the optimal parameter that minimized the MSE.
<img src = "https://rafalab.github.io/dsbook/ml/img/cv-4.png" width = 400 height = 300>

* We can compute the apparent error on the independent validation set. But, 1 sample won't be enough and k should be equal to 5 or 10 since even though larger values of k might seem preferable they take a lot more computational time:
<img src = "https://rafalab.github.io/dsbook/ml/img/cv-5.png" width = 400 height = 400>

Example code of knn:
```r
data("tissue_gene_expression")
# Tissue gene expression data
# Train the data using knn algorithm and ks ranging from 1 to 7 by 2s:
fit <- with(tissue_gene_expression, train(x, y, method = "knn", tuneGrid = data.frame(k = seq(1, 7, 2))))
ggplot(fit)
fit$results
# fit$results this:
 k  Accuracy     Kappa AccuracySD    KappaSD
1 1 0.9932750 0.9918380 0.01078706 0.01310649
2 3 0.9842622 0.9809356 0.01444307 0.01754692
3 5 0.9699980 0.9636815 0.02356627 0.02846193
4 7 0.9617830 0.9537574 0.03046585 0.03686224
```
Another example of knn:
```r
set.seed(1, sample.kind = "Rounding")
library(caret)
# Set the columns of tissue gene expression to variables
y <- tissue_gene_expression$y
x <- tissue_gene_expression$x
# Create an index using data partition on y to ONLY get the outcomes and not the entire datset:
test_index <- createDataPartition(y, list = FALSE)
# Use the created indicies from test_index to create a series of train and test set y's and x's:
train_set_y <- y[-test_index]
test_set_y <- y[test_index]
train_set_x <- x[-test_index, ]
test_set_x <- x[test_index, ]
# Use sapply() to apply to each value of knn, raninging for 1 to 11 with an increment of 2:
accuracies = sapply(seq(1, 11, 2), function(k){
	# Fit the knn model on the trainin set x's and y's with the correct k:
	fit <- knn3(train_set_x, train_set_y, k = k)
	# Predict using the fitted model and using the test_set_x (converted to a data frame)
	y_hat <- predict(fit, as.data.frame(test_set_x), type = "class")
	# Find the mean of y_hat == test_set_y, so your finding the accuracies of each k:
	mean(y_hat == test_set_y)
})
accuracies
# Prints out:
# k value:      1         3        5         7         9         11
# accuracy: 0.9895833 0.9687500 0.9479167 0.9166667 0.9166667 0.9062500
```
* After finding all the MSEs for each k we calculate the average and this gives an estimate of the loss. One more step would be to select the lambdas (parameters) that minimize the MSE. But, the optimization occured on the training data so we need to compute an estimate of the final algorithm based on data that wasn't used to optimize this choice. So, we'll use the test set:
<img src = "https://rafalab.github.io/dsbook/ml/img/cv-6.png" width = 400 height = 300>

* We'll compute cross-validation on the test set but, this won't be for optimization purposes (unlike the training set) but to know the MSE for the final algorithm. However, to calculate the MSE for the final algorithm we need to go through the cross-validation k times:
<img src = "https://rafalab.github.io/dsbook/ml/img/cv-7.png" width = 400 height = 300>

* Once, we're satisfied with our model we can refit the model on the entire data set without changing the parameters:
<img src = "https://rafalab.github.io/dsbook/ml/img/cv-8.png" width = 400 height = 300>

* There is another approach, the *boostrap* approach, which is the default approach in the caret package. It's useful since it can improve the variance of the final estimate by taking more, overlapping samples (picking k sets at some size). Also, this picks observations with replacement which means the same observation can appear twice.
* Suppose the income distribution of the population is:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/income-distribution-1.png" width = 400 height = 300>

* The median of our population is ~ 45,000. Let's say we don't have access to the entire population but want to estimate the median (M). We can take a sample of 250 and estimate the population median with the sample median, which results in ~ 43,000, let's try to build confidence intervals. From a Monte Carlo simulation we can see the sample median is approximatley normal. But, in practice we don't have access to the distribution and the CLT doesn't apply to the median, only averages.
* The *bootstrap* permits us to create a Monte Carlo simulation without access to the entire distribution. We act as if the sample is the entire population and sample, with replacement, data sets of the same size. Then, we can compute the summary statistic (median) on the bootstrap sample. The theory tells us the distribution of the statistic obtained with the bootstrap sample approximates the distribution of the actual statistic (population):
```r
N <- 250
X <- sample(income, N)
M<- median(X)
# Bootstrap monte carlo sim:
B <- 10^4
M_star <- replicate(B, {
    X_star <- sample(X, N, replace = TRUE)
    median(X_star)
})
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/median-is-normal-1.png" width = 400 height = 300>

* The bootstrap sample creates a close estimate of the distribution, providing a decent approximation. Also, we can create the 95% confidence intervals: ```#This creates the actual confidence intervals: quantile(M, c(0.05, 0.95)) #This creates the bootstrap confidence intervals: quantile(M_star, c(0.05, 0.95))```. The confidence intervals from the bootstrap are quite close to the actual ones. This is much better than using the CLT and if we know the distribution is normal we can use the bootstrap to estimate sd, mean, and then confidence interval. Also, the bootstrap is particularly useful in situations in which a tractable variance formula does NOT exist.
* Another bootstrap example of finding the standard error (sd) and expected value (mean) of random, normally distributed data set. When Monte Carlo cannot be run:
```r
# Set the seet to randomly generate a normally distributed data set:
set.seed(1)
y <- rnorm(100, 0, 1)
# Set the seed, again, to create the indicies:
set.seed(1, sample.kind="Rounding")
# Create the 10 bootstrap indexes based on y:
indexes <- createResample(y, 10)
# Apply each index (ind) in indexes to y, then use quantile of 75% to help find the expected value and standard error of 75%
q_75_star <- sapply(indexes, function(ind){
	# Set an an object (y_star) equal to y[ind]
	y_star <- y[ind]
	# Find the quantile of y_star and 0.75
	quantile(y_star, 0.75)
})
# Find the expected value, using mean, which is ~0.731:
mean(q_75_star)
# Find the standard error, using sd, which is ~0.0742:
sd(q_75_star)
# Repeating, the same bootstrap thing with 10,000 bootstrap samples results in the following expected value and standard errror:
# Expected value ~ 0.674
# Standard error ~ 0.0931
```
Generative Models:
* In non-binary circumstances, the conditional probabilites or expectations provide the best approach to developing a decision rule. But, in a binary case the best approach is *Bayes' rule* (p(<strong>x</strong>) = Pr(Y = 1 | <strong>X</strong> = <strong>x</strong>)), which is a decision rule based on the true condtional probability. 
* *Discriminative approaches* estimate the conditional probability directly and do not consider the distribution of the predictors. But, Bayes' theorem tells us knowing the distribution of predictors (<strong>X</strong>) may be useful. *Generative models* are approaches that model the joint distribution of Y (outcomes) and <strong>X</strong> (predictors). Bayes' rule implies that if the distributions of the predictors (conditional distributions) can be estimated then a powerful decision realm can be created. 
* Bayes' rule with f<sub><strong>X</strong>|Y = 1</sub> and f<sub><strong>X</strong>|Y = 0</sub> representing the distribution functions of the predictor <strong>X</strong> for the two classes Y=1 and Y=0:

![p(x) = Pr(Y = 1|X = x) = \frac{f_{X|Y=1}(X)Pr(Y = 1)}{f_{X|Y=0}(X)Pr(Y = 0) + f_{X|Y=1}(X)Pr(Y = 1)}](https://render.githubusercontent.com/render/math?math=p(x)%20%3D%20Pr(Y%20%3D%201%7CX%20%3D%20x)%20%3D%20%5Cfrac%7Bf_%7BX%7CY%3D1%7D(X)Pr(Y%20%3D%201)%7D%7Bf_%7BX%7CY%3D0%7D(X)Pr(Y%20%3D%200)%20%2B%20f_%7BX%7CY%3D1%7D(X)Pr(Y%20%3D%201)%7D)

* We can use the heights dataset for Naive Bayes (generative model) since the conditional distributions of the predictors are normal:
```r
# Generating train and test set
library("caret")
data("heights")
y <- heights$height
set.seed(2)
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
train_set <- heights %>% slice(-test_index)test_set <- heights %>% slice(test_index)

# Estimating averages and standard deviations
params <- train_set %>%
 group_by(sex) %>%
 summarize(avg = mean(height), sd = sd(height))
params

# Estimating the prevalence
pi <- train_set %>% summarize(pi=mean(sex=="Female")) %>% pull(pi)
pi

# Getting an actual rule
x <- test_set$height
f0 <- dnorm(x, params$avg[2], params$sd[2])
f1 <- dnorm(x, params$avg[1], params$sd[1])
# Above finds the distribution of f0 and f1
p_hat_bayes <- f1*pi / (f1*pi + f0*(1 - pi))
```
* Naive Bayes allows us to take into account prevelance (π), the hats (^) will be used to denote our estimates:

![\hat{p}(x) = Pr(Y = 1|X = x) = \frac{\hat{f}_{X|Y=1}(x)\hat{\pi}}{\hat{f}_{X|Y=0}(x)(1-\hat{\pi}) + \hat{f}_{X|Y=1}(x)Pr(Y = 1)}](https://render.githubusercontent.com/render/math?math=%5Chat%7Bp%7D(x)%20%3D%20Pr(Y%20%3D%201%7CX%20%3D%20x)%20%3D%20%5Cfrac%7B%5Chat%7Bf%7D_%7BX%7CY%3D1%7D(x)%5Chat%7B%5Cpi%7D%7D%7B%5Chat%7Bf%7D_%7BX%7CY%3D0%7D(x)(1-%5Chat%7B%5Cpi%7D)%20%2B%20%5Chat%7Bf%7D_%7BX%7CY%3D1%7D(x)Pr(Y%20%3D%201)%7D)

* Our sample has a very small prevelance for females (~23%) which would not work on the population. The low sensitivity can affect the algorithm since it will predict males much more than females. Which can change the prevelance by changing the value of π:
```r
# Changing the Naive Bayes estimate:
p_hat_bayes_unbiased <- f1 * 0.5 / (f1 * 0.5 + f0 * (1 - 0.5))
# Changing the cutoff rule:
y_hat_bayes_unbiased <- ifelse(p_hat_bayes_unbiased > 0.5, "Female", "Male")
```
* Quadratic discriminant analysis (QDA) is a version of Naive Bayes in which we assume that the distributions (f<sub><strong>X</strong>|Y = 1</sub> and f<sub><strong>X</strong>|Y = 0</sub>) are multivariate normal. 
* We can use the digit reader example with 2 digits/outcomes 2s and 7s. We assume the conditional distribution is bivariate normal which is a sub category of multivariate normal disribution. This implies we need to estimate 2 averages, 2 standard deviatsions, and a correlation for each case (2s and 7s). Once we have these we can approximate the conditional distributions f<sub><strong>X</strong><sub>1</sub>, <strong>X</strong><sub>2</sub>|Y = 1</sub> and f<sub><strong>X</strong><sub>1</sub>, <strong>X</strong><sub>2</sub>|Y = 0</sub>:
```r
# Estimate parameters from the data
params <- mnist_27$train %>%
 group_by(y) %>%
 summarize(avg_1 = mean(x_1), avg_2 = mean(x_2),
        sd_1 = sd(x_1), sd_2 = sd(x_2),
        r = cor(x_1, x_2))
params
# y  avg_1  avg_2  sd_1   sd_2    r
# 2  .129   .283   .0702  .0578  .401
# 7  .234   .288   .0719  .105   .455
```
* Visual of both densities (curve encapsulates 95% of the points):
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/qda-explained-1.png" width = 400 height = 300>

* Once these 2 distributions are estimated, this defines an estimate for the conditional probability of Y = 1 | <strong>X</strong><sub>1</sub> & <strong>X</strong><sub>2</sub>. The model can be fitted and used to obtain predictors:
```r
# Fit model
library(caret)
train_qda <- train(y ~., method = "qda", data = mnist_27$train)
# Obtain predictors and accuracy
y_hat <- predict(train_qda, mnist_27$test)
confusionMatrix(data = y_hat, reference = mnist_27$test$y)$overall["Accuracy"]
```
* The estimated conditional probability is pretty close to the true probability. Though, its not good as the fit obtained with kernel smoothers:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/qda-estimate-1.png" width = 400 height = 300>

* One reason why QDA doesn't work as well as the kernel approach, in this case, is because the assumptions of normality don't seem to hold (we assumed a bivariate normal distribution for the 2s and 7s), the 7 seems to be off:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/qda-does-not-fit-1.png" width = 400 height = 300>

* QDA becomes harder to use as the number of predictors increase. In our case, we had 2 predictors and, in total, had to 2 compute 4 means, 4 standard deviations, and 2 correlations. With 10 predictors we have 45 correlations for each class! This formula: K * (2p + p * (p - 1)/2) tells us how many parameters we have to estimate. Once the number of parameters approach the size of the data, the method becomes impractical due to overfitting.
* A solution for having too many parameters is to assume the correlation structure is the same for all classes, reducing the number of parameters needed to estimate. In the 2s and 7s example we would just have to compute 2 means and 2 standard deviations:
```r
params <- mnist_27$train %>%
 group_by(y) %>%
 summarize(avg_1 = mean(x_1), avg_2 = mean(x_2),
        sd_1 = sd(x_1), sd_2 = sd(x_2),
        r = cor(x_1, x_2))
params <- params %>% mutate(sd_1 = mean(sd_1), sd_2 = mean(sd_2), r = mean(r))
params
# y  avg_1  avg_2  sd_1   sd_2    r
# 2  .129   .283   .0710  .0813  .428
# 7  .234   .288   .0710  .0813  .428
```
* Visual of densities for each case. The size of the ellipses and angles are same because of same standard deviation and correlation:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/lda-explained-1.png" width = 400 height = 300>

* When this assumption is forced, mathematically, the boundary is aligned like, logisitic regression. For this reason, the method is called linear discriminant analysis (LDA). Conditional probability with LDA:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/lda-estimate-1.png" width = 400 height = 300>

* The lack of flexibility doesn't allow for a good estimate and the accuracy is quite low, ~ 75%.
* Now, we'll use the same digits datset but, with 1s, 2s, and 7s this time, which can be obtained like this:
```r
if(!exists("mnist"))mnist <- read_mnist()

set.seed(3456)    #use set.seed(3456, sample.kind="Rounding") in R 3.6 or later
index_127 <- sample(which(mnist$train$labels %in% c(1,2,7)), 2000)
y <- mnist$train$labels[index_127] 
x <- mnist$train$images[index_127,]
index_train <- createDataPartition(y, p=0.8, list = FALSE)

# get the quadrants
# temporary object to help figure out the quadrants
row_column <- expand.grid(row=1:28, col=1:28)
upper_left_ind <- which(row_column$col <= 14 & row_column$row <= 14)
lower_right_ind <- which(row_column$col > 14 & row_column$row > 14)

# binarize the values. Above 200 is ink, below is no ink
x <- x > 200 

# cbind proportion of pixels in upper right quadrant and proportion of pixels in lower right quadrant
x <- cbind(rowSums(x[ ,upper_left_ind])/rowSums(x),
           rowSums(x[ ,lower_right_ind])/rowSums(x)) 

train_set <- data.frame(y = factor(y[index_train]),
                        x_1 = x[index_train,1],
                        x_2 = x[index_train,2])

test_set <- data.frame(y = factor(y[-index_train]),
                       x_1 = x[-index_train,1],
                       x_2 = x[-index_train,2])
```
* Graph of predictors and outcomes:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/mnist-27-training-data-1.png" width = 400 height = 300>

* We can fit a QDA model and now we have to estimate 3 conditional probabilites and then predict the digit with the highest probability. The predictors have 3 classes (1s, 2s, and 7s) and for sensitivity and specificity we have a pair of values for each class since to define these terms we need a binary outcome so there are 3 outcomes, one for the corresponding class a positive and the other 2 as negatives:
```r
# Fit the qda model on the dataset:
train_qda <- train(y ~ ., method = "qda", data = train_set)
# Predict the values:
predict(train_qda, test_set, type = "prob") %>% head()
predict(train_qda, test_set) %>% head()
# Accuracy is 75%:
confusionMatrix(predict(train_qda, test_set), test_set$y)$overall["Accuracy"]
```
* We can visualize the estimated conditional probabilities for qda:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/three-classes-plot-1.png" width = 400 height = 300>

* We can do the same visualization for lda and see the accuracy (66%) is much worse because the model is more rigid:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/lda-too-rigid-1.png" width = 400 height = 300>

* The knn does the best as its model is the most flexible and has an accuracy of 77%:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/three-classes-knn-better-1.png" width = 400 height = 300>

* The reason qda and, especially, lda aren't working well is due to lack of fit. Plotting the data shows that, at least, the 1s aren't bivariate normally distributed:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/three-classes-lack-of-fit-1.png" width = 400 height = 300>

* Generative models can be really powerful. But, only when the join distribution of predictors each class can be succesfully approximated.
* I'll show you another example of a generative model, with lda, we're trying to predict the correct tissue type (y) based on a sample of 10 other tissues:
```r
# Load up the proper libraries, data, and set the seed:
library(dslabs)      
library(caret)
data("tissue_gene_expression")
set.seed(1993, sample.kind="Rounding") 
# Set the y column of tissue_gene_expression to a variable (y), which are the outcomes:
y <- tissue_gene_expression$y
# Set the x column of tissue_gene_expression to a variable (x), which are the predictors:
x <- tissue_gene_expression$x
# Take a sample of 10 from x:
x <- x[, sample(ncol(x), 10)]
# Fit an lda model on x and y. Also, center/scale each column using the preProcess argument:
fit_lda <- train(x, y, method = "lda", preProcess = c("center"))
# Find the accuracy, which is ~ 82%
fit_lda$results["Accuracy"]
```
<strong>Classification With More Than 2 Classes And The Caret Package:</strong>
* LDA and QDA aren't meant to  be used with datasets that have too many predictors, since the number of parameters to estimate becomes too large. For example, for the digits example, which has 784 predictors, lda would have to estimate over 600,000 parameters and for qda you would have to mulitply that by the number of classes (10), which results in 6,000,000 parameters! Kernel methods, like k nearest neighbor, or local regression don't have model parameters to estimate and they face the *curse of dimensionality* when multiple predictors are used. The dimension refers to the fact that when we have p predictors the distance between 2 observations calculated in p dimensional space.
* One way to understand the *curse of dimensionality* is by considering how large we have the make the neighborhood of estimates to include a certain percentage of the data, with large neighborhoods the methods loose flexibility. Assume we have 1 continous predictor with equally space points in the [0, 1] interval and want to create a window that includes 10% of the data, then our window has to be of size, 0.1:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/curse-of-dim-1.png" width = 400 height = 100>

* For 2 predictors if we want to include 0.1 of each dimension then it would be a single point. 10% of the whole dataset would require each side of the square to be sqrt(10) and would include 0.316 of the whole data:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/curse-of-dim-2-1.png" width = 400 height = 200>

* To include 10% of the data with p dimensions we need an interval with each side having a size of p√.1 or .1<sup>1/p</sup>. This grows very quickly and gets close to 1, which means all the data, without smoothing:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/curse-of-dim-4-1.png" width = 400 height = 300>

* We'll use a dataset that includes the breakdown of olives into 8 fatty acids, ```data("olive")```. We'll try to predict the region/location (Northern Italy, Sardina, or Southern Italy) of olive using the fatty acid composition values as predictors. The area column is removed because it's not used as a predictor, ```olive = select(olive, -area)```. Let's see how good we do using knn:
```r
# Predict region using KNN
library(caret)
fit <- train(region ~ .,  method = "knn", 
             tuneGrid = data.frame(k = seq(1, 15, 2)), 
             data = olive)
# We get an accuracy of 97%.
ggplot(fit)
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/olive-knn-1.png" width = 300 height = 200>

* However, using data exploration, we see we can do even better. Looking at the distribution of each predictor stratified by region we see that eicosenoic is only present in Southern Italy and that linoleic separates Northern Italy from Sardinia. Also, as explained, before a goot k for knn would be 5 or 10 and the accuracy for that drops to 95%.

Code: 
```r
olive %>% gather(fatty_acid, percentage, -region) %>%
  ggplot(aes(region, percentage, fill = region)) +
  geom_boxplot() +
  facet_wrap(~fatty_acid, scales = "free", ncol = 4) +
  theme(axis.text.x = element_blank(), legend.position="bottom")
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/olive-eda-1.png" width = 300 height = 200>

* By looking at this we can construct a prediction rule for eicosenoic and linoleic.

Code:
```r
olive %>% 
  ggplot(aes(eicosenoic, linoleic, color = region)) + 
  geom_point() +
  geom_vline(xintercept = 0.065, lty = 2) + 
  geom_segment(x = -0.2, y = 10.54, xend = 0.065, yend = 10.54, 
               color = "black", lty = 2)
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/olive-two-predictors-1.png" width = 300 height = 200>

* The decision rule can be if the first predictor (eicosenoic) is larger than 0.065 predict Southern Italy. If not, then if the second predictor (linoleic) is larger than 10.535 then predict Sardina. Otherwise, predict Northern Italy. This can be drawn up as a *decision tree*, like so:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/olive-tree-1.png" width = 300 height = 200>

* *Decision trees* are used widely in practice, like a decision tree that a docotor uses for deciding if a person is at risk for a heart attack:
<img src = "https://rafalab.github.io/dsbook/ml/img/Decision-Tree-for-Heart-Attack-Victim-adapted-from-Gigerenzer-et-al-1999-4.png" width = 300 height = 200>

* The general idea is to define algorithms that use data to create trees, like the ones shown. Regression and decision trees operate by predicting an outcome variable, y, by partitioning preditor space. When the outcome is continous the algorithms are called *regression trees*.
* We'll use the poll data from 2008, trying to estimate the conditional expectation of y (poll margin) given x (day), f(x) = E(Y | <strong>X</strong> = <strong>x</strong>).
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/polls-2008-again-1.png" width = 300 height = 200>

* So we have to build a decision tree and each node will have a different prediction (Ŷ). To do this, we partition the predictor space (J) into non-overlapping regions (R<sub>1</sub>, R<sub>2</sub>, ..., R<sub>J</sub>). For every observation that falls within a region (x<sub>i</sub> ∈ R<sub>i</sub>) predict Ŷ with the average of the training observations Y<sub>i</sub> in the region.
* Regression trees create the paritions (R<sub>1</sub>, R<sub>2</sub>, ..., R<sub>J</sub>) *recursively*, the resursive steps will be exaplined. Assume we already have a partition, so that every observation (i) is in exactly one of these partitions. For each of these partitions we'll divde further using the following algorithm:
	1. We need to find a predictor (j) and a value (s) that define 2 new partitions (R<sub>1</sub> and R<sub>2</sub>). These 2 paritions will split our observations into the following sets: R<sub>1</sub>(j, s) = {<strong>X</strong> | X<sub>j</sub> < s} and R<sub>2</sub>(j, s) = {<strong>X</strong> | X<sub>j</sub> < s}.
	2. Then, in each of these sets we'll define an average, ỸR<sub>1</sub> for R<sub>1</sub> and ỸR<sub>2</sub> for R<sub>2</sub> and use these as our predictions. The averages will be the averages of the observations in each of the 2 partitionss. 
	3. We could do this for many j's and s' but we pick the combinations that minimize the RSS (residual sum of squares).
	4. Then, the whole recursion is applied recursively, new regions to split in 2 kept on getting found.
* Once, the paritioning of predictor space into regions is completed, then, in each region, a prediction is made using that region, just calculate an average.
* We can see what this algorithm does on the 2008 poll data, the rpart function in the rpart package will be used: ```fit <- rpart(margin ~ ., data = polls_2008)```. Since there's only 1 predictor we don't need to decide which predictor (j) to split by. We just need to decide which values (s) we need to split, we can visually see were the splits were made:

Code: 
```r
plot(fit, margin = 0.1)
text(fit, cex = 0.75)
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/polls-2008-tree-1.png" width = 300 height = 200>

* The 1st split is made on day 39.5, then 1 of those 2 regions is split at day 86.5. The resulting 2 partitions are split on days 49.5 and 117.5, respectively. In the end, there are 8 partitions. The final estimate looks like this:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/polls-2008-tree-fit-1.png" width = 300 height = 200>

Code:
```r
polls_2008 %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col="red")
```
* Every time we split and define 2 new paritions the training set residual sum of squares decreases, because with more partitions the model has more flexibility to adapt to the training data. In fact, splitting until every point is its own partition the RSS goes down to 0, since the average of 1 value is the same value.
* To avoid this overtraining the algorithm sets a minimum on how much the RSS must improve for another partition to be added, this parameter is referred to as the *complexity parameter (cp)*. The RSS must improve by a factor of cp for the new partition to be added.
* Also, the algorithm sets a minimum number of observations to be partitioned, the rpart function has an argument called minsplit that let's you define this, with the default being 20. The algorithm sets a minimum on the number of observations in each partition, in the rpart function the argument is called minbucket (if the optimal split results in a number of observations less than minbucket it's not considered), defaulting to the rounded value of minsplit divided by 3 (minbucket = round(minsplit/3)).
* Let's see what happens when cp is set to 0 and minsplit to 2. The prediction will be the training data since the tree will keep on splitting and splitting until the RSS is 0, resulting in heavy overtraining:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/polls-2008-tree-over-fit-1.png" width = 300 height = 200>

* We can, also, *prune* trees by snipping off partitions that don't meet a cp criterion: ```pruned_fit <- prune(fit, cp = 0.01)```. We can grow a tree very large and then prune off branches to make it smaller. The resulting estimate:
<img src = "https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/pruned_tree.PNG" width = 300 height = 200>

* Cross-validation can be used to pick the best cp value. The train() function can be used for this:
```r
# use cross validation to choose cp
library(caret)
train_rpart <- train(margin ~ ., 
			method = "rpart", 
			tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)), 
			data = polls_2008)
ggplot(train_rpart)
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/polls-2008-tree-train-1.png" width = 350 height = 250>

* The tree that minimizes the MSE (mean squared error) can be accessed via the finalModel component:
```r
# access the final model and plot it
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)
polls_2008 %>% 
  mutate(y_hat = predict(train_rpart)) %>% 
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col="red")
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/polls-2008-final-model-1.png" width = 350 height = 250>

Since there's only 1 predictor f(x) can be plotted:

<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/polls-2008-final-fit-1.png" width = 350 height = 250>

* When the outcome is categorical these methods are referred to as classification or decision trees. The same basic principles are used as with the continuous case but, some slight changes are made to account for the fact the data is categorical.
* Differences:
	* One difference is, rather than taking the average at the end of each node for choosing which class to predict. The class that appears the most in a node is predicted.
	* We can't use RSS to decide on a partition since the outcomes are categorical. We could a naive approach, like looking for partitions that minimize training error. But, better performing approaches use more sophisticated methods. 2 of these are the *Gini Index* and *Entropy*. We can define p̂<sub>m, k</sub> as the proportion of observations in partition m that are of class k.
		1. Then, the Gini Index is defined as: Gini = <sup>K</sup>∑<sub>k = 1</sub> p̂<sub>m, k</sub>(1 - p̂<sub>m, k</sub>)
		2. Entropy is defined as: Entropy = -<sup>K</sup>∑<sub>k = 1</sub> p̂<sub>m, k</sub> log(p̂<sub>m, k</sub>), with 0 * log(0) defined as 0.
	* Both of the above metrics seek to partition observations into subsets that have the same class, they want *purity*. Note: If a partion (m) has only 1 class (1st one) then p̂<sub>m, 1</sub> = 1, p̂<sub>m, 2</sub> = 0, ..., p̂<sub>m, K</sub> = 0. When this happens, both Gini index and Entropy are 0.
* Let's see how classification trees perform on the digits dataset, consiting of 2s and 7s:
```r
train_rpart <- train(y ~ .,
              method = "rpart",
              tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
              data = mnist_27$train)
plot(train_rpart)
```
We can pick the best cp from this plot:

<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/mnist-27-tree-1.png" width = 300 height = 200>

* We can use that cp with the tree and see how well we do: ```confusionMatrix(predict(train_rpart, mnist_27$test), mnist_27$test$y)$overall["Accuracy"]```. An accuracy of 82% is achieved, this is better than logistic regression but not as good as the kernel methods. The limitations of a classification tree are shown here (with decision trees the boundary cannot be smoothed):
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/rf-cond-prob-1.png" width = 350 height = 350>

* Despite these limitations classification trees have some advantages that make them very useful. They're highly interpretable, more so than linear models (linear regression or logistic regression models). Also, they're easy to visualize, if small enough. And, they, sometimes, model human decision processes. On the other hand, the recursive partioning approach is a bit harder to train than like, linear regression or k-nearest neighbors. Also, it's not very flexible so it might not be the best performing method and is quite susceptible to changes in the training data. Random forests can improve on these shortcomings.
* *Random forests* try to increase prediction performance and reduce instability by averaging multiple decision trees, creating a forest of decision trees made with randomness. It has 2 features for this.
	1. One feature is *bootstrap aggregation* or *bagging*. For this we build many decision or regression trees, T<sub>1</sub>, T<sub>2<sub>, ..., T<sub>B</sub> using the training set. The bootstrap is used for this, to create B bootstrap trees we create tree (T<sub>j</sub, j = 1, ..., B) from a training set of size N. But, to create T<sub>j</sub> we create a bootstrap training set by sampling N observations from the training set with replacement. Then, a decision tree is built for each bootstrap training set, this keeps the indivual decision trees random.
	2. Next, for every observation (j) in the test set we form a prediction (ŷ<sub>j</sub>) using the corresponding tree (T<sub>j</sub>). To obtain a final prediction we combine the the prediction for each tree in 2 different ways, for continuous outcomes and categorical outcomes. For continuous outcomes we take the average of predictions (ŷ<sub>j</sub>). On the other hand, for categorical outcomes the ŷ that appears the most is predicted, the class that appears the most.
* Applying random forest to the 2008 poll data and plotting error versus number of trees:
```r
library(randomForest)
fit <- randomForest(margin~., data = polls_2008) 
plot(fit)	
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/more-trees-better-fit-1.png" width = 300 height = 200>
	
* As the number of trees increases, the error goes down, goes up a bit, and then levels out. But, more complex problems will require more trees for the algorithm to converge. The final result for the poll 2008 data:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/polls-2008-rf-fit-1.png" width = 300 height = 200>

* The final result is somewhat smooth and not a step function like the indivual trees, the averaging allows us to permit estimates that aren't step functions. As the number of trees grow, the step function becomes less abundant because of the averaging:
<img src = "https://rafalab.github.io/dsbook/ml/img/rf.gif" width = 300 height = 200>

* We can fit a random forest to our 2s and 7s digit example:
```r
library(randomForest)
train_rf <- randomForest(y ~ ., data=mnist_27$train)
confusionMatrix(predict(train_rf, mnist_27$test), mnist_27$test$y)$overall["Accuracy"]
# Returning an accuracy of 80%
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/cond-prob-rf-1.png" width = 400 height = 300>

* We have much more flexibility than a single tree but, this particular random forest is a big too wiggly. We can use the caret package to optimize some parameters and make it smoother. We can use a different random forest algorithm, Rborist, which is a bit faster:
```r
# use cross validation to choose parameter
train_rf_2 <- train(y ~ .,
      method = "Rborist",
      tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
      data = mnist_27$train)
confusionMatrix(predict(train_rf_2, mnist_27$test), mnist_27$test$y)$overall["Accuracy"]
# Accuracy is 80.5%
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/cond-prob-final-rf-1.png" width = 400 height = 200>

* There are several ways to control the smoothness of the random forest estimate. One way is to limit the size of each node, requiring the number of points per node to be larger (minsplit). Also, we can use a random selection of features to split partitions. Specifically, when building each tree at each recursive partition, we only consider a randomly selected subset of predictors to check for the best split and every tree has a different random selection of features. This reduces correlation between trees in the forest, which, in turn, improves prediction accuracy. The argument for this tuning parameter in the random forest function is mtry but each ranomd forest implementation has a different name, looking at the help file to figure out which one. 
* A disadvantage of random forest is we lose interpretability, we're averaging hundres or thousands of trees into a forest. However, there's a measure called *variable importance* that helps us interpret the results, it tells us how much each predictor influences the final predictions.

<strong>Caret Package:</strong>
* The algorithms we've learned so far (logistic regression, knn, random forests, and etc) are a small subset of all the algorithms out there. Many of these algorithms are implemented in R but, they're distributed across a wide array of packages, developed by different authors, and use different syntax.
* The caret package tries to consolidate these algorithms and provide consistency and contains more than 200 different methods that are summarize in this [site](http://topepo.github.io/caret/available-models.html). Caret doesn't automatically install the packages needed to run these methods so to implement a package through caret you still need to install the library. The required packages for each method is included [here](http://topepo.github.io/caret/train-models-by-tag.html).
* The caret package, also, provides a function that performs cross-validation.
* We'll use the 2s and 7s digit data set to show some ways in which the caret package can be used. The train() function allows us to train different algorithms using similar syntax, like we can train logistic regression model or knn model just by specifying the model argumnet:
```r
library(tidyverse)
library(dslabs)
data("mnist_27")
library(caret)
# Train logistic regression:
train_glm <- train(y ~ ., method = "glm", data = mnist_27$train)
# Train knn:
train_knn <- train(y ~ ., method = "knn", data = mnist_27$train)
```
* To make the predictions we can use the output of train() directly without looking at the specific of predict.glm() or predict.knn(), we can look at predict.train() and read the help page:
```r
# Prediction for logistic regression:
y_hat_glm <- predict(train_glm, mnist_27$test, type = "raw")
# Prediction for knn:
y_hat_knn <- predict(train_knn, mnist_27$test, type = "raw")
```
* We can study the accuracies very quickly:
```r
confusionMatrix(y_hat_glm, mnist_27$test$y)$overall[["Accuracy"]]
# Returns accuracy of 75% for logistic regression.
confusionMatrix(y_hat_knn, mnist_27$test$y)$overall[["Accuracy"]]
# Returns accuracy of 84% for logisitc regression.
```
* When an algorithm contains a tuning parameter, train() automatically conducts a cross-validation to decide among a few default values. To find out what parameter(s) are optimized, read [this](https://topepo.github.io/caret/available-models.html). Or, the getModelInfo() and modelLookup() functions can be used to learn more about a model and the parameters that can be optimized: ```modelLookup("knn")```. We run the train() function with default values and see what parameter(s) is/are optimized, using cross-validation:
```r
# Running train() with default values:
train_knn <- train(y ~ ., method = "knn", data = mnist_27$train)
# Plotting the function, hilighting the parameter the optimizes the algorithm:
ggplot(train_knn, highlight = TRUE)
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/caret-highlight-1.png" width = 300 height = 200>

* By default, the cross-validation is performed by testing on 25 bootstrap samples comprised of 25% of the observations. For the knn method the default is to try out 5, 7, and 9 with 9 maximizing this but there might be an even better k. To change this we need to use the tunegrid parameter in the train() function. The grid of values that are going to be compared must be supplied with a data frame with the column names as specified by the parameters that you get in the model lookup output. 
* Let's try out 30 values between 9 and 67. We need to use a column in k so the data frame will be: ```data.frame(k = seq(9, 67, 2))```. When running this code we're fitting 30 versions of knn to 25 bootstrap samples, totaling to 750 knn models which will take several seconds:
```r
# Train the 750 knn models:
train_knn <- train(y ~ ., method = "knn", 
                   data = mnist_27$train,
                   tuneGrid = data.frame(k = seq(9, 71, 2)))
# Plot the values of k versus accuracy:
ggplot(train_knn, highlight = TRUE)
# Returns the k that maximizes accuracy, k = 29 and accuracy = ~ 85%:
train_knn$bestTune
# Best performing model (29-nearest neighbor model) is accesed using the below code:
train_knn$finalModel
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/train-knn-plot-1.png" width = 300 height = 200>

* The k that maximizes accuracy is 29 and the accuracy = ~ 85%. If you apply a predict() functionto the output of the train() function then it'll use the best performing model to make predictions. Note: The best model was obtained using only the training set (cross-validaton performed on training set) so now we can see the accuracy obtained on the test set: 
```r
confusionMatrix(predict(train_knn, mnist_27$test, type = "raw"),
+               mnist_27$test$y)$overall["Accuracy"]
# Accuracy is 0.835, ~ 84%.
```
* Sometimes we might change the way we perform cross-validation, the method, the way how partitions are made, etc. For this we need to use the trainControl() function, we can make the code, just shown, go a bit faster by using 10-fold cross-validation:
```r
# 10 validation samples that use 10% of the observations, each:
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn_cv <- train(y ~ ., method = "knn", 
+                     data = mnist_27$train,
+                     tuneGrid = data.frame(k = seq(9, 71, 2)),
+                     trControl = control)
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/cv-10-fold-accuracy-estimate-1.png" width = 300 height = 200>

* The accuracy estimates are more variable than without the 10-fold cross-validation. This happens since we changed the number of samples used to estimate accuracy, in the first example we used 25 bootstrap samples, and now we used 10-fold cross-validation.
* The train() function provides standard deviation values for each parameter that was tested, obtained from the different validation sets. We can make a plot that shows the point estimates of accuracy with standard deviations:
<img src https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/accuracies_with_sd.PNG"" width = 300 height = 200>

* The best fitting knn model approximates the true conditional probability pretty well. But the boundary is a bit wiggly since knn, like the basic bin smoother, does not use a smooth kernel:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/mnist27-optimal-knn-fit-1.png" width = 300 height = 200>

* To improve this we could try loess, scanning through the [avaliable models page](https://topepo.github.io/caret/available-models.html) we find tht we can use the gamLoess() method. We see that we need to install the gam package. By using: ```modelLookup("gam")``` we can see that we have 2 parameters to optimize. For this we'll keep the degree fixed at 1. But, to try out different values for the span we would still have to include a column in the table with the name degree, being a requirment of the caret package, so we can define a grid using expand.grid(): ```grid <- expand.grid(span = seq(0.15, 0.65, len = 10), degree = 1)```. Now, we use the default cross-validation control parameters:
```r
# Training model:
train_loess <- train(y ~ ., 
+              method = "gamLoess",
+              tuneGrid=grid,
+              data = mnist_27$train)
# Plot all models on accuracy vs span:
ggplot(train_loess, highlight = TRUE)
# Select the best performing model based on accuracy which is ~0.32 with an accuracy of 85%:
confusionMatrix(data = predict(train_loess, mnist_27$test), 
                reference = mnist_27$test$y)$overall["Accuracy"]
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/loess-accuracy-1.png" width = 300 height = 200>

* The best performing model had a span of ~0.32 and an accuracy of 85%, performing similar to the best performing model of knn. But, the conditional probability estimate is much smoother than knn's:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/gam-smooth-1.png" width = 300 height = 200>

* Not all parameters in machine learning algorithms are tuned. For example, in reggression models or lda (linear discriminant analysis) we fit the best model using least squares estimate or maximum likelihood estimates, which aren't tuning parameters, they're obtained using least squares or MLE (maximum likelihood estimate), or some other optimization technique. Parameters that are turned are parameters that we can change and then get an estimate of the model for each one. So in knn the number of neighbors is a tuning parameter, in regression the number of predictors that we include could be considered a parameter that's been optimized. Thus, in the train() function of the caret package we only optimize parameters that are tunable. And, the train() function in the caret package won't optimize the regression coefficents that are estimated, instead it'll just estimate using least squares.
* It's very important to make a clear distinction to make when using the caret package, knowing which parameters are optimized and which aren't.
* Example code for finding best value of cp for a classification tree fit, changing nodesize to 0, plotting best tree fit, creating a random forest:
```r
# Load the proper libraries:
library(caret)
library(rpart)          
library(dslabs)
data("tissue_gene_expression")
set.seed(1991, sample.kind = "Rounding")
# Change the type of tissue_gene_expression:
dat = as.data.frame(tissue_gene_expression)
y = dat$y
# Create a classification tree fit with "rpart" for different values of cp:
fit_classification_tree = train(y ~., method = "rpart", tuneGrid = data.frame(cp = seq(0, 0.1, 0.01)), data = dat)
# Plot the fit to see which value cp is th best, cp = 0 gives the highest accuracy ~ 89%:
ggplot(fit_classification_tree)
set.seed(1991, sample.kind = "Rounding")
# Change minsplit = 0 so any node can be split (makes it less flexible but increases accuracy) since ther are
# only 6 placentas in the dataset:
fit_rpart = train(y ~., method = "rpart", tuneGrid = data.frame(cp = seq(0, 0.1, 0.01)), data = dat, control = rpart.control(minsplit = 0))
# Access the accuracy when cp = 0 via confusionMatrix, accuracy = ~ 91%:
confusionMatrix(fit_rpart)
# Plot the tree with this highest accuracy:
plot(fit_rpart$finalModel)
text(fit_rpart$finalModel)
set.seed(1991, sample.kind = "Rounding")
# Create a random forest (method = "rf"), find the best mtry value, and set nodesize to 0:
fit = train(y ~., method = "rf", tuneGrid = data.frame(mtry = seq(50, 200, 25)), data = dat, minsplit = 1)
# Access the best mtry value (100):
fit$bestTune$mtry
# Set the output of train() with random forest ("rf") to a function varImp() stored in the variable, imp:
imp = varImp(fit)
# Print out importantce of each variable in the random forest method, stored in imp:
imp
#rf variable importance
# only 10 most important variables shown (out of 500)
#         Overall
#GPA33     100.00
#BIN1       64.65
#GPM6B      62.35
#KIF2C      62.15
#CLIP3      52.09
#COLGALT2   46.48
#CFHR4      35.03
#SHANK2     34.90
#TFR2       33.61
#GALNT11    30.70
```

<strong>Model Fitting and Recommendation Systems:</strong>

Case Study (MNIST):
* We're going to use the MNIST (Modified National Institute of Standards and Technology database) digits data set, its a popular dataset among machine learning competitions. Which be loaded like this: ```mnist = read_mnist()```. The data has training and test sets: ```names(mnist)```. We'll sample 10,000 random rows from the training set and 1,000 random rows from test set:
```r
# sample 10k rows from training set, 1k rows from test set
set.seed(123)
index <- sample(nrow(mnist$train$images), 10000)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])
index <- sample(nrow(mnist$test$images), 1000)
x_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])
```
* In machine learning, predictors are often transformed before running the machine learning algorithm and predictors that are unnessecary are removed, this is referred to as *preprocessing*. 
* Examples of preprocessing include standarizing the predictors, taking the transform (like log) of some predictors, removing predictors that are highly correlated with each other, and removing predictors with very few non-unique values or close to 0 variation.
* We can see there are a lot of features with 0 variability or almost 0 variability, using this:
```r
library(matrixStats)
# Compute the sd of each column:
sds <- colSds(x)
# Plot it:
qplot(sds, bins = 256, color = I(“black”))
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/pixel-sds-1.png" width = 300 height = 200>

* This is expected since there are parts of the image that contain very few dark pixels, very litte writing and very little variation. Almost all the values are 0. The caret package includes a function that recommends these features to be removed because of *near zero variance*:
```r
library(caret)
nzv <- nearZeroVar(x)
image(matrix(1:784 %in% nzv, 28, 28))
# The columns that are removed are the purple ones in the plot
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/near-zero-image-1.png" width = 300 height = 200>

* Once the columns are removed we're left with 252 columns: ```col_index <- setdiff(1:ncol(x), nzv)```.
* Now, we're going to actually implement knn and random forest on the MNIST data but, before we do this we need to add column names to the feature matricies, being a requirment of the caret package:
```r
colnames(x) <- 1:ncol(mnist$train$images)
# The name of the column will be its colum number:
colnames(x_test) <- colnames(x)
```
* We can start with trying to create a knn model:

	1. The first step would be to optimize for the number of neighbors. When we run the algorithm we'll have to compute the distance between each observation in the test set and each observation in the training set, resulting in a lot of calculations. The k-fold cross-validation can be used to improve speed:
```r
# The control contains the 10-fold cross validation with 10% chance.
control <- trainControl(method = "cv", number = 10, p = .9)
# We can find the model that maximixes accuracy
train_knn <- train(x[,col_index], y,
                                method = "knn", 
                                tuneGrid = data.frame(k = c(1,3,5,7)),
                                trControl = control)
```
* The above code might take several minutes to run on a standard laptop. In general, it's a good idea to take a small subset of the data to test out the piece of code and get a sense of its timing before we start running code that might take hours to run, or even days. We can test the above code on a smaller dataset with n as the number of rows and b as the number of cross-validations folds:
```r
# Number of rows/observations:
n <- 1000
# Number of cross-validation folds with 10% chance:
b <- 2
index <- sample(nrow(x), n)
control <- trainControl(method = "cv", number = b, p = .9)
train_knn <- train(x[index ,col_index], y[index],
                   method = "knn",
                   tuneGrid = data.frame(k = c(3,5,7)),
                   trControl = control)
```
* We can keep on increasing n and b to get an idea of how long the whole process will take. Once we're done optimizing the algorithm we can fit the entire data: ```fit_knn <- knn3(x[ ,col_index], y,  k = 3)```. The accuracy is almost 95%:
```r
# Predict using fitted model (knn) on test set:
y_hat_knn <- predict(fit_knn,
                     x_test[, col_index],
                     type="class")
# Create confusion matrix:
cm <- confusionMatrix(y_hat_knn, factor(y_test))
# Find accuracy:
cm$overall["Accuracy"]
```
* We can obtain the sensitivty and specificity outputs:
```r
cm$byClass[,1:2]
#>          Sensitivity Specificity
#> Class: 0       0.990       0.996
#> Class: 1       1.000       0.993
#> Class: 2       0.965       0.997
#> Class: 3       0.950       0.999
#> Class: 4       0.930       0.997
#> Class: 5       0.921       0.993
#> Class: 6       0.977       0.996
#> Class: 7       0.956       0.989
#> Class: 8       0.887       0.999
#> Class: 9       0.951       0.990
```
* From the sensitivty and specificity outputs we see the 8s are the hardest to detect and the most commonly incorrect predicted digit is 7.

	2. Let's try out random forest and see if we can do even better. Though with random forests computation time is an even bigger challenge than with knn. For each forest we need to build hundreds of trees and there are several parameters we can tune. We use the random forest implementation in the Rborist package (but it has less features) which is faster than the rf package. Since the fitting the is the slowest part of the procedure, instead of the predicting (which is with knn), for random forest, we'll use only use 5-fold cross-validation. Also, we'll reduce the number of trees in the fit since we're not building the final model, kind of like taking a subset. Lastly, we'll take a random subset of observations when constructing each tree, this can be changed with the nSamp argument in the Rborist() function:
```r
library(Rborist)
# Set control as the 5-fold cross-validation with 20% chance for each one:
control <- trainControl(method="cv", number = 5, p = 0.8)
# Set grid as the tuning parameters
grid <- expand.grid(minNode = c(1,5) , predFixed = c(10, 15, 25, 35, 50))
# Create the random forest model with Rborist:
train_rf <-  train(x[, col_index], y,
                   method = "Rborist",
                   nTree = 50,
                   trControl = control,
                   tuneGrid = grid,
                   nSamp = 5000)
```
* The above optimization takes a few minutes to run. We can choose the best parameters using this: ```train_rf$bestTune```. And, now we're ready to set our final tree, so we'll increase the number of trees:
```r
fit_rf <- Rborist(x[, col_index], y,
                  nTree = 1000,
                  minNode = train_rf$bestTune$minNode,
                  predFixed = train_rf$bestTune$predFixed)
```
* The above code takes a few minutes to run. But, once it's done we can use the below code to find the accuracy:
```r
# Set y_hat_rf to class of factor:
y_hat_rf <- factor(levels(y)[predict(fit_rf, x_test[ ,col_index])$yPred])
# Create confusion matrix of the above and y_test:
cm <- confusionMatrix(y_hat_rf, y_test)
# Print accuracy (95.1%):
cm$overall["Accuracy"]
```
* We've done minimal tuning here and with some more tuning, examining more parameters, growing out more trees, we can get an even higher accuracy.
* One of the limitations, as explained before, of random forests is they're not very interpretable. However the concept of *variable importance* helps a little bit in this regard. Unfortuantley, the current implementation of the Rborist package doesn't support variable importance calculations. So we can use the random forest function in the random forest package, instead. And, we're not going to use all the columns in the feature matrix:
```r
library(randomForest)
# Sets x and y and creates a random forest model:
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])
rf <- randomForest(x, y,  ntree = 50)
```
* We can compute the importance of each feature now: ```imp = importance(rf)```. The first few features have 0s because they're never used in the importance algorithm because they're on the edges. In this particular example it makes sense to explore the importance of these features via an image, we can make an image in which each feature is plotted on where it came from in the image, ```image(matrix(imp, 28, 28))```:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/importance-image-1.png" width = 300 height = 200>

* The important features are in the middle, where most of the writing is. An important part of data science is visualizing results to discern why we're failing, the method of doing this depending on the application. For the digits we'll find digits which we were quite certain of a call but it was incorrect. We can seee some incorrect calls for random forest:
```r
# NOTE: THE BELOW CODE IS 4 KNN FALSE POSITIVE CALLS:
# Predict using knn:
p_max <- predict(fit_knn, x_test[,col_index])
p_max <- apply(p_max, 1, max)
# Find the indicies which don't match up with the test data:
ind  <- which(y_hat_knn != y_test)
# Order it in decreasing order:
ind <- ind[order(p_max[ind], decreasing = TRUE)]
```
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/rf-images,-1.png" width = 500 height = 200>

* We can see where we messed up with random forest, up above, and learn how to fix our algorithm.
* A very powerful approach in machine learning is the idea of ensembling different machine learning algorithms into one. The idea of an *ensemble* is similar to the idea of combining data from different pollsters to obtain a better estimate of the true support for different candidates. In machine learning, one can greatly improve the final results of our prediction by combining the results of different algorithms.
* We can compute new class probabilites by combining the class probabilities of knn and random forest. We can use the below code, to simply average these probabilities:
```r
# Random forest:
p_rf <- predict(fit_rf, x_test[,col_index])$census
p_rf <- p_rf / rowSums(p_rf)
# knn:
p_knn <- predict(fit_knn, x_test[,col_index])
p <- (p_rf + p_knn)/2
# Combine both probailities:
y_pred <- factor(apply(p, 1, which.max)-1)
# Create a confusion matrix and print it:
confusionMatrix(y_pred, y_test)
```
* The ensemble has a probability of 96.1% which is greater than knn (94.9%) and random forest (95.4%). We only ensemble 2 methods but in practice we might ensemble dozens or hundreds of methods, which really provides some substansial improvements.
