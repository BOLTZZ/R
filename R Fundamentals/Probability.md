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
# One more way is to use the mean function.
mean(x < 70) # Finds the probability of a male shorter than 70 inches.
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

Statistial Inference:
* Statistical inference offers a framework for quantifying uncertainty due to randomness.

Randomness, Sampling Models, and Centeral Limit Theorem:
* A sampling model models the random behavior of a process as the sampling of draws from a dataset.
* The probability distribution of a random variable is the probability of the observed value falling in any given interval. The interval can be denoted in the CDF function (F(a) = Pr(x≤a)) as x.
* The average of many draws of a random variable is called its expected value.
* The standard deviation of many draws of a random variable is called its standard error.
* The difference between probability distribution and distribution is that any list of numbers has distribution. The probability distribution function of a random variable is defined mathematically and does not depend on a list of numbers. But, the probability distribution can be found using a list of numbers (average and standard deviation) using the CDF function.
* Capital letters denote random variables (X) and lowercase letters denote observed values (x).
* In the notation Pr(X=x), we are asking how frequently the random variable X is equal to the value x.
* The Centeral Limit Theorem (CLT) states that when the independent draws (sample size) is large, the probability distribution of the sum of the independent draws is approximately normal. The sample size required for the Central Limit Theorem and Law of Large Numbers to apply differs based on the probability of success. If the probability of success is high, then relatively few observations are needed. As the probability of success decreases, more observations are needed. But, sometimes the probability of success can be extremley low, like winning the lottery. The formula for CLT in R is to use pnorm().
* The expected value is the average of the values in a dataset, which, in turn, represents the value of one draw. The formula for the expected value is E[X] = (summation of all values)/(count of all values). Note: the equation to the left considers that every value in the datset has an EQUAL CHANCE of being pulled. Also, expected value is denoted by E[X].
* The equation for the expected value with 2 variables and different proprtions (UNEQUAL CHANCE) is: ap + b * (1−p). With a and b the 2 variables/values, p the proproption/chance of a, and 1-p the proportion/chance of b. Also, with n draws the equation for the sum of n is: n * (ap + b * (1−p)). Meanwhile, for the average of n is: ap + b * (1−p)
* The standard error (SE) gives the size of the variation around the expected value, since the expected value is basically and estimation there will be variations in each case/observation. 
* If the draws are independent then the SE is S[X] = sqrt(number of draws) * (standard deviation of the numbers in the dataset). S[X] denotes the standard error.
* The SE for 2 variables (a and b) with different proprotions/chance (a has p and b has 1 - p) is: ∣b–a∣ * sqrt(p * (1 - p)). And, then with n draws for the sum of n it's: sqrt(n) * ∣b–a∣ * sqrt(p * (1 - p)). But, for the average of n it's: (|b-a| * sqrt(p * (1 - p)))/sqrt(n).

Example code for expected value with 1 draw, n draws, standard error with 1 draw, n draws:
```r
# The variables 'green', 'black', and 'red' contain the number of pockets for each color
green <- 2
black <- 18
red <- 18
# Assign a variable `p_green` as the probability of the ball landing in a green pocket
p_green <- green / (green+black+red)
# Assign a variable `p_not_green` as the probability of the ball not landing in a green pocket
p_not_green <- 1-p_green
# Calculate the expected outcome if you win $17 if the ball lands on green and you lose $1 if the ball doesn't land on green
expected_value_with_1_draw = (p_green * 17) + (-1 * p_not_green)
# Compute the standard error of the random variable
standard_error_with_1_draw = abs(17 - -1) * sqrt(p_green * p_not_green)
# Define the number of bets using the variable 'n'
n = 1000
# Create a vector called 'X' that contains the outcomes of 1000 samples
X = sample(c(-1, 17), 1000, replace = TRUE, prob = c(p_not_green, p_green))
# Calculate the expected outcome of 1,000 spins if you win $17 when the ball lands on green and you lose $1 when the ball doesn't land on green
expected_value_with_1000_draws = 1000 * ((17 * p_green) + (-1 * p_not_green))
# Compute the standard error of the sum of 1,000 outcomes
standard_error_with_1000_draws = sqrt(1000) * abs(17 - -1) *  sqrt(p_green * p_not_green)
# Compute the standard error of the average of 1,000 outcomes
standard_error_average = abs(17 - -1) *  sqrt(p_green * p_not_green)/sqrt(1000)
```
Properties:
1. The expected value of the sum of random variables (the dataset) is the sum of the expected values of indivual random variables.
2. The expected value of a random variable * non-random constant is that variable's expected value * non-random constant. E[aX] = a * E[X]
* Because of the above 2 properties, the expected value of the average of the draws from a dataset is the expected value of the dataset.
3. The square of the SE of the sum of the independent random variables is the sum of the square of the SE for each random variable.
4. The SE of a random variable * non-random constant is the SE * non-random constant.
* Because of the above 2 properties, the SE of the average independent draws from the same dataset is the standard deviation of the dataset divide by the square root of n (SE = sd(dataset)/sqrt(n))
5. If X is a normally distributed randon variable and a and b are non-random constants then, a * X + b is also a normally distributed ranomd variable.

  Law of Large Numbers or the Law of Averages:

  As the number of n draws increases the SE becomes very small (since there is more data to estimate on). Once, n is very large the SE is practically 0 and the average of the draws converges to the average of the dataset.

* The CLT only applies to only approximately normaly distributed datasets. Also, the number of values needed can vary to only 10 to as large as 1 million. And, the chance of succes can't be that small for the CLT (to use CLT in R use pnorm()). The Poisson distribution would be better in these cases.
