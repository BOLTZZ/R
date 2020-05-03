*Note: Check out [data science functions](https://github.com/BOLTZZ/R/blob/master/R%20Fundamentals/R%20Basics%20&%20Syntax.md#import-data-gathering-functions) for data science related functions.*
# Data Inference & Modeling:
Parameters and Estimates:
* take_poll(number_you_want) can be used to take a sample.
* The spread can be modeled by 2p - 1 (p - (1 - p)) where p is the proportion of one variable in a dataset. Note: The dataset which I'm saying is called population and the proprotion of one variable in the population (p) is called a parameter.
* Statistical inference, the process of deducing characteristics of a population using data from a random sample. Also, the goal of statistical inference is to predict the parameter (p) using the observed data in the sample.
* Many common data science tasks can be framed as estimating a parameter from a sample.
* Forecasting takes into account that p can change over time and create tools to better predict while keeping in mind that p might change. While, polling doesn't take change over time in account because it doesn't need to.
* Standard error (SE), in polling, including sample size is: S[random_variable] = sqrt(sample_size * p * (p - 1)). Also, expected value is: E[random_variable] sample_size * p. When solving the expected value for the average, the formula is: E[sample_average] = p. Also, the SE for the average is: S[sample_average] = sqrt((p * (1 - p))/sample_size).
Centeral Theorem Limit in Practice:
* The margin of error is 2 times the standard errror.
* Since p is unknown its hard to run Monte Carlo simulations on CLT. Though, there is a way to run Monte Carlo simulations on CLT by setting p to a value, or setting multiple values, before running the simulation.:
```r
p <- 0.45    # unknown p to estimate
N <- 1000

# simulate one poll of size N and determine x_hat
x <- sample(c(0,1), size = N, replace = TRUE, prob = c(1-p, p))
x_hat <- mean(x)

# simulate B polls of size N and determine average x_hat
B <- 10000    # number of replicates
N <- 1000    # sample size per replicate
x_hat <- replicate(B, {
    x <- sample(c(0,1), size = N, replace = TRUE, prob = c(1-p, p))
    mean(x)
})
```
* Bias is errors in polling. For example, an extremely large poll would theoretically be able to predict election results almost perfectly. But, there is an issue of cost, people might lie to you, you'd be missing out on people who don't have a phone but still vote, and how do you know that everyone in your population will vote. Typical bias is usally from 1 to 2%.
* You can use qnorm() to find probability of a specific value in random variable (qnorm(specific_value, expected_value, standard_error)):
```r
# Define `p` as the proportion of Democrats in the population being polled
p <- 0.45

# Define `N` as the number of people polled
N <- 100

# Calculate the probability that the estimated proportion of Democrats in the population is greater than 0.5. Print this value to the console.
standard_error = sqrt(p * (1 - p)/N)
1 - pnorm(0.5, p,standard_error)
# Note: p is the expected value and SE is standard_error
```
Condfidence Intervals & p-values:
* Confidence intervals are intervals that are likely to include p with a certain degree of confidence. The intervals can't be too big (like from 0 to 1) and they have to be accurate. 
* To calculate any size confidence interval, we need to calculate the value z for which Pr(−z ≤ Z ≤ z) equals the desired confidence. For example, a 99% confidence interval requires calculating z for Pr(−z ≤ Z ≤ z) = 0.99.
* For a confidence interval of size q, we solve for z = 1 − (1 − q) / 2.
* To determine a 95% confidence interval, use z <- qnorm(0.975). This value is slightly smaller than 2 times the standard error. The lower bound of the 95% confidence interval is equal to: p − qnorm(0.975) ∗ standard_error_of_p. The upper bound of the 95% confidence interval is equal to  p + qnorm(0.975) ∗ standard_error_of_p .
* Monte Carlo simulations can be ran to test confidence intervals (this one tests a 95% confidence interval):
```r
B <- 10000
inside <- replicate(B, {
    X <- sample(c(0,1), size = N, replace = TRUE, prob = c(1-p, p))
    X_hat <- mean(X)
    SE_hat <- sqrt(X_hat*(1-X_hat)/N)
    between(p, X_hat - 2*SE_hat, X_hat + 2*SE_hat)    # TRUE if p in confidence interval
})
mean(inside)
# Note mean(inside) returned ~0.95 or 95% the confidence interval was correct. Like it should be!
# Also, The 95% confidence intervals are random, but  p  is not random. And, 95% refers to the probability that the random interval falls on top of  p .
```
* Power is the probability of detecting an effect when there is a true effect to find (detecting a spread different from 0). Power increases as sample size increases, because larger sample size means smaller standard error.
* The null hypothesis is the hypothesis that there is no effect.
* The The p-value is the probability of detecting an effect of a certain size or larger when the null hypothesis is true.
* We can convert the probability of seeing an observed value under the null hypothesis into a standard normal random variable. We compute the value of z that corresponds to the observed result, and then use that z to compute the p-value. Also, if a 95% confidence interval doesn't include our observed value we can conclude the p-value must be less than 0.05 because 1 - 95% = 0.05.
* It is preferable to report confidence intervals instead of p-values, as confidence intervals give information about the size of the estimate and p-values do not.
* To find p-values for a given z-score z in a normal distribution with mean mu and standard deviation sigma, use 2*(1-pnorm(z, mu, sigma)) instead. If the mean = 0 and s.d. = 1 of a normal distribution, use 2*(1-pnorm(2)) instead.
* 
