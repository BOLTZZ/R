# Data Visualization:
There are 3 types of data:
1. Categorical (like male or female).
2. Numerical (numbers).
3. Ordinal is categorical with a scale.

2 types of data point:
1. Continous (like time).
2. Discrete (like number of humans).

Distribution:
* Distribution is how distributed the data is.
* Frequency tables are used to model distribution.
* Distribution is not good for numerical data so cumulative distribution function (CDF) is used.
* CDF shows a proportion of the data below a value a for all possible values of a. 
* The mathematicl notation is F(a) = Pr(x <= a)
A function, F(a), is made equal to the proportion of values x less than or equal to a.
* CDF is not very popular in practice because it doesn't give a lot of key info. easily (distribution symmetric?, center?, and etc). Histograms are much easier to intrepet.
CDF can be calculated and plotted like this:
```r
a <- seq(min(my_data), max(my_data), length = 100)    # define range of values spanning the dataset
cdf_function <- function(x) {    # computes prob. for a single value
    mean(my_data <= x)
}
cdf_values <- sapply(a, cdf_function)
plot(a, cdf_values)
```
* Decreasing the bin size of histograms can create a smooth density plot. But, we should have very precise data to make the bin size small otherwise the data on the smooth desnity plot won't be accurate.
* Normal distribution is centered around the mean, symmetrical around the mean, and can be represented using the mean and standard deviation.
* The z score is the number of S.Ds a data point is from the mean. The z score can be used to find the percentile.
