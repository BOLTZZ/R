*Note: Check out [data science functions](https://github.com/BOLTZZ/R/blob/master/R%20Fundamentals/R%20Basics%20&%20Syntax.md#import-data-gathering-functions) for data science related functions.*
# Data Inference & Modeling:
Parameters and Estimates:
* take_poll(number_you_want) can be used to take a sample.
* The spread can be modeled by 2p - 1 (p - (1 - p)) where p is the proportion of one variable in a dataset. Note: The dataset which I'm saying is called population and the proprotion of one variable in the population (p) is called a parameter.
* Statistical inference, the process of deducing characteristics of a population using data from a random sample. Also, the goal of statistical inference is to predict the parameter (p) using the observed data in the sample.
* Many common data science tasks can be framed as estimating a parameter from a sample.
* Forecasting takes into account that p can change over time and create tools to better predict while keeping in mind that p might change. While, polling doesn't take change over time in account because it doesn't need to.
* Standard error (SE), in polling, including sample size is: S[random_variable] = sqrt(sample_size * p * (p - 1)). Also, expected value is: E[random_variable] sample_size * p. When solving the expected value for the average, the formula is: E[sample_average] = p. Also, the SE for the average is: S[sample_average] = sqrt((p * (1 - p))/sample_size). While population average is just the average of the population and population standard deviation is the sd of the population.
* Use sample() to sample from a population:
```r
# The vector of all male heights in our population `x` has already been loaded for you. You can examine the first six elements using `head`.
head(x)

# Use the `set.seed` function to make sure your answer matches the expected result after random sampling
set.seed(1)

# Define `N` as the number of people measured
N <- 50

# Define `X` as a random sample from our population `x`
X = sample(x, N, replace = TRUE)

# Calculate the sample average. Print this value to the console.
mean(X)

# Calculate the sample standard deviation. Print this value to the console.
sd(X)
```
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
* Two-tailed test:
```r
# We made an object `res` to summarize the average, standard deviation, and number of polls for the two pollsters.
res <- polls %>% group_by(pollster) %>% 
  summarize(avg = mean(spread), s = sd(spread), N = n()) 

# The variables `estimate` and `se_hat` contain the spread estimates and standard error, respectively.
estimate <- res$avg[2] - res$avg[1]
se_hat <- sqrt(res$s[2]^2/res$N[2] + res$s[1]^2/res$N[1])

# Calculate the p-value
( 1 - pnorm(estimate/se_hat)) * 2
```
Statistical Models:
* Poll aggregators combine the results of many polls to simulate polls with a large sample size and therefore generate more precise estimates than individual polls.
* Pollster bias reflects the fact that repeated polls by a given pollster have an expected value different from the actual spread and different from other pollsters. Each pollster has a different bias. We need to create a data driven model for a better estimate and confidence interval to account for bias.
* In the data driven model for pollsters we have 2 unknown quantaties, expected value, d,  and standard deviation, sigma. Though, the CLT still works for this model since its an average of independent, random variables. If the sample size, N, is large enough then the probability distribution of the sample average will be approximately normal with expected value, d, and standard deviation, sigma/sqrt(N). The standard deviation can be estimated with the sample standard deviation and the sd() computes the sample standard deviation. Since d is a fixed parameter we can't calculate probabilites, yet, for this we will need to learn about Bayesian statistics.

Bayesian Statistics:
* In the earlier model, we discussed where the probability, p, is a fixed number (it can't change). But, in Bayesian statistics the probability is random. Hierarchical models describe variability at different levels and incorporate all these levels into a model for estimating p.
* Bayes theorem is used to find the probability of event A happening given event B is equal to the probability of both A and B divided by the probability of event B (REMEMBER: the | operator stands or given that):

Pr(A∣B) = (Pr(B∣A) * Pr(A))/Pr(B) = Pr(A and B)/Pr(B)

* The techniques we have used up until now are referred to as frequentist statistics as they consider only the frequency of outcomes in a dataset and do not include any outside information. Frequentist statistics allow us to compute confidence intervals and p-values.
* Frequentist statistics can have problems when sample sizes are small and when the data are extreme compared to historical results.
* Bayesian statistics allows prior knowledge to modify observed results, which alters our conclusions about event probabilities.
* Hierarchical models use multiple levels of variability to model results. They are hierarchical because values in the lower levels of the model are computed using values from higher levels of the model.
* In the Bayesian model we set up the mean, μ and standard deviation, τ based on past data. Like if we were trying to predict the popular vote winner (Democrat or Republican) for Florida then the mean (mu) would be close to 0 because past data has shown its been really close for Democrat and Republican and the s.d. (tau) would be ~ 0.01 because of past data.
* We model baseball player batting average using a hierarchical model with two levels of variability:
    * p ∼ N * (μ,τ) describes player-to-player variability in natural ability to hit, which has a mean μ and standard deviation τ.
    * Y ∣ p ∼ N * (p,σ) describes a player's observed batting average given their ability p, which has a mean p and standard deviation  σ = p * (1−p)/sqrt(N). This represents variability due to luck.
* In Bayesian hierarchical models, the first level is called the prior distribution and the second level is called the sampling distribution. The posterior distribution allows us to compute the probability distribution of p given that we have observed data Y.
* By the continuous version of Bayes' rule, the expected value of the posterior distribution p given Y=y is a weighted average between the prior mean μ and the observed data Y:

E(p ∣ y) = B * μ + (1 − B) * Y = μ + (1 − B) * (Y - μ)

and B = σ^2 / σ^2 + τ^2 with the standard error as SE(p∣Y)^2 = 1/(1/σ^2+1/τ^2). Note that you will need to take the square root of both sides to solve for the standard error.

* This Bayesian approach is also known as shrinking. When σ is large, B is close to 1 and our prediction of p shrinks towards the mean μ. When σ is small, B is close to 0 and our prediction of p is more weighted towards the observed data Y.

Some example code:
```r
# Load the libraries and poll data
library(dplyr)
library(dslabs)
data(polls_us_election_2016)

# Create an object `polls` that contains the spread of predictions for each candidate in Florida during the last polling days
polls <- polls_us_election_2016 %>% 
  filter(state == "Florida" & enddate >= "2016-11-04" ) %>% 
  mutate(spread = rawpoll_clinton/100 - rawpoll_trump/100)

# Examine the `polls` object using the `head` function
head(polls)

# Create an object called `results` that has two columns containing the average spread (`avg`) and the standard error (`se`). Print the results to the console.
results <- polls %>% summarize(avg = mean(spread),  se = sd(spread)/sqrt(n()))
# Define `mu` and `tau`
mu <- 0
tau <- 0.01

# Define a variable called `sigma` that contains the standard error in the object `results`
sigma = results$se
# Define a variable called `Y` that contains the average in the object `results`
Y = results$avg
# Define a variable `B` using `sigma` and `tau`.
B = sigma^2 / (sigma^2 + tau^2)
# Calculate the expected value of the posterior distribution
exp_value = mu + (1 - B) * (Y - mu)
# Compute the standard error of the posterior distribution. Print this value to the console.
se = sqrt(1/((1/sigma^2) + (1/tau^2)))

# Construct the 95% credible interval. Save the lower and then the upper confidence interval to a variable called `ci`.
ci <- c(B*mu + (1-B)*Y - qnorm(0.975)*sqrt( 1/ (1/sigma^2 + 1/tau^2)), B*mu + (1-B)*Y + qnorm(0.975)*sqrt( 1/ (1/sigma^2 + 1/tau^2)))

# Using the `pnorm` function, calculate the probability that the actual spread was less than 0 (in Trump's favor). Print this value to the console.
pnorm(0, exp_value, se)
# Returned value = 0.3203769
```
Election Forecasting:
* The spread d ∼ N(μ,τ) describes our best guess in the absence of polling data. We set μ=0 and τ=0.035 using historical data.
* The average of observed data X ∣d ∼ N(d,σ) describes randomness due to sampling and the pollster effect.
* Because the posterior distributon is normal, we can report a 95% credible interval that has a 95% chance of overlapping the parameter using E(p∣Y) and SE(p∣Y).
* Given an estimate of E(p∣Y) and SE(p∣Y), we can use pnorm to compute the probability that  d>0 .
* Pollsters have this general bias (either for Democrat or Republican) that can greatly sway our predictions. But, this bias can be combated with the equation below.
* If we collect data from a pollster and assume it has no bias then there's a theory that tells us the random polls collected (X<sub>1</sub>, ..., X<sub>J</sub>) of sample size N have an expected value of d and standard error of 2 * sqrt(p * (1 - p)/N) for the spread. 
* We represent each measurement as X<sub>i,j</sub> = d + b + h<sub>i</sub> + ϵ<sub>i,j</sub>  where:
    * The index i represents the different pollsters
    * The index j represents the different polls
    * X<sub>i,j</sub> is the jth poll by the ith pollster 
    * d is the actual spread of the election
    * b is the general bias affecting all pollsters
    * hi represents the house effect for the ith pollster (pollster variability)
    * ϵ<sub>i,j</sub> represents the random error associated with the i,jth poll.
```r
I <- 5
J <- 6
N <- 2000
d <- .021
p <- (d+1)/2
h <- rnorm(I, 0, 0.025)    # assume standard error of pollster-to-pollster variability is 0.025
X <- sapply(1:I, function(i){
    d + rnorm(J, 0, 2*sqrt(p*(1-p)/N))
})
```
* The sample average changes to sample_average = d + b + (1/N) <sup>N</sup>∑<sub>i = 1</sub> * sample_average<sub>i</sub>. Also, the sample standard deviation changes to sample_standard_deviation = sqrt(σ^2/N + σ^2<sub>b</sub>)
* The standard error of the general bias σ<sub>b</sub> does not get reduced by averaging multiple polls, which increases the variability of our final estimate.

Code (of probability d>0 with general bias):
```r
mu <- 0
tau <- 0.035
sigma <- sqrt(results$se^2 + .025^2)
Y <- results$avg
B <- sigma^2 / (sigma^2 + tau^2)

posterior_mean <- B*mu + (1-B)*Y
posterior_se <- sqrt(1 / (1/sigma^2 + 1/tau^2))

1 - pnorm(0, posterior_mean, posterior_se)
```
* Forecasters need to realize that pollstes don't take into account variability over time which is why our model needs to include a bias term (b<sub>t</sub>) to model the time effect. Also, f(t) estitmates the trend of p given at t, which is modeled as: Y<sub>i,j,t</sub> = d + b + h<sub>j</sub> + b<sub>t</sub> + f(t) + ϵ<sub>i,j,t</sub>.
* left_join can join 2 dataframes together:
```r
# Add the actual results to the `cis` data set
add <- results_us_election_2016 %>% mutate(actual_spread = clinton/100 - trump/100) %>% select(state, actual_spread)
ci_data <- cis %>% mutate(state = as.character(state)) %>% left_join(add, by = "state")
```
* Since we introduce further variability into our confidence intervals with this new model resulting in a confidence interval that is "overconfident". In very large sample sizes this extra variability doesn't matter but, with sample sizes equal to or less than ~ 30 we need to be catious about using CLT. If the population data follows the normal distribution then we can use a mathematical theory that tells us how much bigger we need to make the intervals to account for the estitmated sigma. Though, NOTE: the following theory is robust for datasets with deviations from normal distribution (not too much deviation, though).
* This was the equation for CLIT: Z = (sample_average - d)/(sigma/sqrt(N)) which changes because we estimate sigma now (say sigma is now s): Z = (sample_average - d)/(s/sqrt(N)). Z, now, follows a t-distribution with N - 1 degrees of freedom. The degrees of freedom control variability with things called fatter tails (weight of tails of normal distribution). The smaller the degrees of freedom are (the fatter the tails are) the increased chance of extreme values.
* The confidence intervals can be determined using t-distribution instead of normal distribution with the funciton qt():
```r
z <- qt(0.975, nrow(one_poll_per_pollster) - 1)
one_poll_per_pollster %>%
    summarize(avg = mean(spread), moe = z*sd(spread)/sqrt(length(spread))) %>%
    mutate(start = avg - moe, end = avg + moe)

# quantile from t-distribution versus normal distribution
qt(0.975, 14)    # 14 = nrow(one_poll_per_pollster) - 1 <- Degrees of freedom. Also, its calculating 95% confidence intervals.
qnorm(0.975)
```
Association Tests:
* There is a way to determine the probability an observation is due to random variability given categorical, ordinal, or binary data. It's called Fisher's test.
* Fisher's exact test determines the p-value as the probability of observing an outcome as extreme or more extreme than the observed outcome given the null distribution.
* Data from a binary experiment are often summarized in two-by-two tables. The p-value can be calculated from a two-by-two table using Fisher's exact test with the function fisher.test():
```r
# Lady Tasting Tea Problem:
tab <- matrix(c(3,1,1,3), 2, 2)
rownames(tab) <- c("Poured Before", "Poured After")
colnames(tab) <- c("Guessed Before", "Guessed After")
tab

# p-value calculation with Fisher's Exact Test
fisher.test(tab, alternative = "greater"
```
* In 2 by 2 tables the sums of the rows and columns are fixed which allow for the hypergoemtric distribution. But, in general the data isn't limited so we use the chi-squared test. The chi-squared test compares the observed two-by-two table to the two-by-two table expected by the null hypothesis and asks how likely it is that we see a deviation as large as observed or larger by chance. It uses an asymptotic result similar to what is used by CLT. The function chisq.test() takes a two-by-two table and returns the p-value from the chi-squared test: 
```r
# compute overall funding rate
funding_rate <- totals %>%
    summarize(percent_total = (yes_men + yes_women) / (yes_men + no_men + yes_women + no_women)) %>%
    .$percent_total
funding_rate

# construct two-by-two table for observed data
two_by_two <- tibble(awarded = c("no", "yes"),
                    men = c(totals$no_men, totals$yes_men),
                     women = c(totals$no_women, totals$yes_women))
two_by_two

# compute null hypothesis two-by-two table
tibble(awarded = c("no", "yes"),
           men = (totals$no_men + totals$yes_men) * c(1-funding_rate, funding_rate),
           women = (totals$no_women + totals$yes_women) * c(1-funding_rate, funding_rate))

# chi-squared test
chisq_test <- two_by_two %>%
    select(-awarded) %>%
nbsp;   chisq.test()
chisq_test$p.value
```
* An informative summary statistics associated with the 2 by 2 tables is the odds ratio. The odds ratio states how many times larger the odds of an outcome are for one group relative to another group.:
```r
# odds of getting funding for men
odds_men <- (two_by_two$men[2] / sum(two_by_two$men)) /
        (two_by_two$men[1] / sum(two_by_two$men))

# odds of getting funding for women
odds_women <- (two_by_two$women[2] / sum(two_by_two$women)) /
        (two_by_two$women[1] / sum(two_by_two$women))

# odds ratio - how many times larger odds are for men than women
odds_men/odds_wome
```
* The relationship between p-value and odds ratio is not 1 to 1, it depends on the sample size. Also, a small p-value does not imply a large odds ratio. If a finding has a small p-value but also a small odds ratio, it may not be a practically significant or scientifically significant finding:
```r
# multiplying all observations by 10 decreases p-value without changing odds ratio
two_by_two %>%
  select(-awarded) %>%
  mutate(men = men*10, women = women*10) %>%
  chisq.test()
```
* The odds ratio is a ratio of ratios, there is no simple way to use the Central Limit Theorem to compute confidence intervals (though, there are some complex ways). One approach is to use the theory of generalized linear models. 
