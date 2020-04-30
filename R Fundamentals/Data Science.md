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
Quantiles, Percentiles, and Quartiles:
* Quantiles are points that divide the dataset into intervals with set probabilities. The xth quantile is at which x% of the observations are equal to or less than the x value.
* Percentiles divide up the dataset to a 1% probability (100 intervals).
* Quartiles divide up the dataset into 4 intervals, each with a 25% probability, that are equal to 25% (1st quartile), 50% (median), and 75% (3rd quartile). 
* The summary() function returns the minimum, quartiles, and maximum of a vector.
```r
# Quantile with qth quantile of data:
quantile(data, q)
# All percentiles of the set below:
p = seq(0.01, 0.99, 0.01)
percentiles = quantile(data, p)
# Finding quartiles (we can use the percentiles up above):
percentiles[names(percentiles) == "25%"]
percentiles[names(percentiles) == "50%"]
percentiles[names(percentiles) == "75%"]
```
* qnorm() function gives the theoretical value of a quantile with probability p of observing a value equal to or less than that quantile value given a normal distribution with mean mu and standard deviation sigma:
```r
qnorm(p, mu/mean, sigma/standard_deviation)
```
* pnorm() function gives the probability that a value from a standard normal distribution will be less than or equal to a z-score value z.
```r
pnorm(-1.96)  ≈0.025 
# The result of pnorm() is the quantile. Note that:
qnorm(0.025)  ≈−1.96 
# qnorm() and pnorm() are inverse functions:
pnorm(qnorm(0.025))  =0.025
```
* Plotting theoretical quantiles and observed quantiles (QQ plot):
```r
data(heights)
index <- heights$sex=="Male"
x <- heights$height[index]
# calculate observed and theoretical quantiles:
p <- seq(0.05, 0.95, 0.05)
observed_quantiles <- quantile(x, p)
theoretical_quantiles <- qnorm(p, mean = mean(x), sd = sd(x))
# make QQ-plot:
plot(theoretical_quantiles, observed_quantiles)
abline(0,1)
# make QQ-plot with scaled values:
z <- scale(x)
observed_quantiles <- quantile(z, p)
theoretical_quantiles <- qnorm(p)
plot(theoretical_quantiles, observed_quantiles)
abline(0,1)
```
* Sometimes normal distribution doesn't fit the data so we can't give a 2 number summary of the mean/median ± standard deviation. So we can give a 5 number summary (max., min., L.Q., U.Q., and median) using a boxplot. 
