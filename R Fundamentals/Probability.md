*Note: Check out [data science functions](https://github.com/BOLTZZ/R/blob/master/R%20Fundamentals/R%20Basics%20&%20Syntax.md#import-data-gathering-functions) for data science related functions.*
# Probability:
Discrete Probability:
* Categorical data uses discrete probability.
* Probability of an event = Proportion of times the event occurs when we repeat the experiment over and over, independently and under the same cirumstances:

Pr(A) = Probability of event A
* Event is defined as an outcome that can happen by chance. 
* sample(vector, num_of_items_to_pick) can be used to randomly pick item(s), based on the value of num_of_items_to_pick, from the vector.

  Monte Carlo Simulations:
  * Monte Carlo simulation is when you repeat an experiment so many times the outcomes are similar to if it was being repeated       infinitely. You can change the random seed of sample() using set_seed() and make sure to set it to a number before doing Monte Carlo. Also, Monte Carlo simulations can be really accurate if the repetitions are sufficent.
  * replicate(num_of_times_to_be_repeated, function_to_be_repeated) can emulate a Monte Carlo simulation. Set num_of_times_to_be_repeated to a very high number (10,000). Also, function_to_be_repeated is usually sample().
  * To find which number makes a Monte Carlo simulation stable (number that makes the probability accurate) you can keep on making the number of repitions larger and larger and graph this until there are no more wide flucuations:
  
![1](https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/monte_carlo.PNG)
  
* Probability distribution is the proportion for each group.
* Conditional probabilites are useful when events are non-independent (dependent) using the multiplicative rule:

Pr(A and B) = Pr(A) * Pr(B | A) 

Note: The pipe operator (|) stands for given that or conditional or. This can also be expanded to hold more variables:

Pr(A and B and C) = Pr(A) * Pr(B | A) * Pr(C | A and B)

The multiplicative rule for independent events is:

Pr(A and B) = Pr(A) * Pr(B) 

OR

Pr(A and B and C) = Pr(A) * Pr(B) * Pr(C)
* The function, expand.grid(), gives all the combinations for 2 vectors.
```r
expand.grid(pants = c("blue", "black"), shirts = c("white", "grey", "plaid"))
```
* permutations(n, r, dataset) from the gtools package lists the different ways that r items can be selected from a set of n options when order matters.
* combinations(n, r, datset) from the gtools package lists the different ways that r items can be selected from a set of n options when order does not matter.
```r
sapply(vector, operation) # sapply allows element-wise operations on vectors.
```
* duplicated() and any() are important functions for probability. duplicated() checks for duplications and any() checks if any value is equal to a certain one:
```r
set.seed(1)
celtic_wins <- replicate(10000, {
  simulated_games <- sample(c("lose","win"), 4, replace = TRUE, prob = c(0.6, 0.4))
  simulated_games
  any(simulated_games == "win")
})
```
Addition rule - The addition rule is the probability of A or B:

Pr(A or B) = Pr(A) + Pr(B) − Pr(A and B)

Continous Probability:
* The cumulative distribution function (CDF) is standard for assiging intervals for continous probability. The CDF is a distribution function for continuous data x that reports the proportion of the data below a for all values of a:

F(a) = Pr(x≤a)

* An example CDF:
```r
F <- function(a) mean(x <= a)
1 - F(70)    # probability of male taller than 70 inches
# Another way to create this is using pnorm():
avg = mean(x)
standard_deviation = sd(x)
pnorm(70, avg, standard_deviation) # probability of a male shorter than 70 inches.
```
* The cumulative distribution for normal distribution data in R is pnorm():
```r
F(a) = pnorm(a, avg, sd)
# A random quantity is normall distributed with an average of avg and standard deviation of sd.
# So we can use the above equation for normally distributed data like height.
# NOTE: pnorm() is only for continous data.
```
* The quantile for normal distribution in R is qnorm():
```r
qnorm(quantile, avg, sd) # returns the data at that quantile.
# Example (male heights at 99th percentile):
99th_percentile = qnorm(0.99, mean(x), sd(x))
```
Probability Density:
* An integral can be thought of as the area under the curve up to the value of a gives you the probability of x being less than or equal to the value of a.
* The probability density  f(x)  is defined such that the integral of  f(x)  over a range gives the CDF of that range:

F(a)=Pr(X≤a)=∫a−∞f(x)dx

* The probability density function of a normal distribution is given by dnorm(). Also, pnorm() gives the distribution function, which is the integral of the density function.  
```r
dnorm(z) # gives the probability density f(z) of a certain z-score
dnorm(z, mu, sigma)
```
* rnorm(n, avg, s) generates n random numbers from the normal distribution with average avg and standard deviation s. This allows us to create data that mimics normal distribution.
* rnorm() allows us to simulate Monte Carlo simulations:
```r
B <- 10000
tallest <- replicate(B, {
    simulated_data <- rnorm(800, avg, s)    # generate 800 normally distributed random heights
    max(simulated_data)    # determine the tallest height
})
mean(tallest >= 7*12)    # proportion of times that tallest person exceeded 7 feet (84 inches)
```
* Other distributions are sutdent-t, chi-squared, exponential, gamma, beta, etc.
* R provides functions for density (d), quantile (q), probability distribution (p) and random number generation (r) for many of these distributions.
* Each distribution has a matching abbreviation (for example, norm() or t()) that is paired with the related function abbreviations (d, p, q, r) to create appropriate functions. So, this create pNORM(), dNORM(), qNORM(), and etc (sorry of the emphasis).
