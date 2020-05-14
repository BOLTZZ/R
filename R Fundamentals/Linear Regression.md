*Note: Check out [data science functions](https://github.com/BOLTZZ/R/blob/master/R%20Fundamentals/R%20Basics%20&%20Syntax.md#import-data-gathering-functions) for data science related functions.*
# Linear Regression:
* Linear regression is commonly used to quantify the relationship between two or more variables. It is also used to adjust for confounding. 

<strong>Regression Overview:</strong>
* Bill James was the originator of the sabermetrics, the approach of using data to predict what outcomes best predicted if a team would win.
* There is chance involved in scoring runs in baseball because sometimes hitting it too hard can make it catchable if a catcher is in the right position. This chance makes it good for data analysis.
* The batting averages (hits/# of times at bat) is considered one of the most important offensive statistic. But, this ignores bases on balls which is a success rate.
* Do teams with more home runs score more runs? We can use the Lahman library and graph this as a scatterplot because of the 2 variables. There is a strong, positive correlation showing that teams with more home runs score more runs. We can graph bases on balls and runs and we, also, see a pretty strong, positive correlation. Does this mean bases on balls cause more runs? Probably not, it appears bases on balls are causing runs but home runs are causing both (bases on balls and runs). This is called *confounding*.

Correlation:
* The correlation coefficent summarizes how 2 variables (like father heights and son heights) move together. The correlation coefficient is defined for a list of pairs (x<sub>1</sub>,y<sub>1</sub>),...,(x<sub>n</sub>,y<sub>n</sub>) as the product of the standardized values: ((x<sub>i</sub>−μ<sub>x</sub>)/σ<sub>x</sub>) * ((y<sub>i</sub>−μ<sub>y</sub>)/σ<sub>y</sub>). The correlation coefficient is always between -1 and 1, with 0 being unrelated variables or no correlation. The cor() finds the correlation for 2 variables: ```cor(variable_1, variable_2)```
* In data science, we mostly don't observe the population but a sample of the population. Sample correlation is used for population correlation just like, pop. average -> sample average, pop. sd -> sample sd. This implies, the sample correlation we use is a random variable and can have a high standard of error. Since the sample correlation is an average of independent draws, the Centeral Limit Theorem (CLT) still applies.

Stratification And Variation:
* Correlation is not always a good summary between 2 variables. An example of this is Anscombe's quartet, some artificial data:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/ascombe-quartet-1.png" width = 300 height = 200>
All the graphs above have a correlation of 0.82 but look extremley different when graphed.

* Assume we have to guess the height of a randomly selected son, we would pick the average (70.5) of the sample size since heights are normally distributed. The son's father has a height of 72 inches, which is 1.14 sd above average for father heights. Should we pick the son's height as 1.14 sd above the mean, just like the height of the father. No, this would be an overestimate, looking at all the sons with fathers of 72 inches heights. This is accomplished by *stratifying* the dataset into only fathers with 72 inches heights. What we're doing is called a *conditional average*, average son height *conditioned* on the father being 72 inches tall.
* The challenge with stratifying and conditional averages are that variables being exactly the certain value we're looking for can be very small, resulting in large standard errors. This can be combated in our dataset by creating a *strata* of very similar heights by rounding the heights to the nearest inch and finding 72 from there. This results in the son's height being 71.8 inches (being 0.54 sd taller than the average son's height). 
* Once, the each son conditional average based on each strata of father heights are graphed on a box plot. The centers/means (son's heights) are increasing with father's heights:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/boxplot-1-1.png" width = 200 height = 200>

* The means of each group seem to follow a linear relationship. In fact, the slope of the line for these points is ~0.5 which is the correlation coefficent between father and son heights. This line is called the regression line:
<img src = "https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/regression_line.PNG" width = 200 height = 200>

* The regression line for 2 variables, x and y, says that for every standard deviation (σ<sub>x</sub>) increase above the average (μ<sub>x</sub>), y grows ρ (correlation coefficent) standard deviations (σ<sub>y</sub>) above the average (μ<sub>y</sub>). Formula: ((y<sub>i</sub> - μ<sub>y</sub>)/σ<sub>y</sub>) = ρ * ((x<sub>i</sub> - μ<sub>x</sub>)/σ<sub>x</sub>). 
* For perfect correlation we predict increase that is the same number of sd. If there is no correlation we don't use x at all for the prediction of y. Values between 0 and 1, prediciton is somewhere in between. Negative correlation then predict a reduction instead of increase.
* When the correlation is positive but smaller than 1 that we predict something closer to the mean is called regression. The son *regresses* to the average height.
* When you know the change of height from the mean for the father and want to find that change of height for the son, just find the slop of the regression line and mulitply it by the change of the father. For example, father is 1 inch above the mean what is the predicited change for the son (ρ = 0.5, σ<sub>son</sub> = 3, σ<sub>father</sub> = 2). Slope = 0.5 * (3/2) = 0.75 and 0.75 * 1 = 0.75, so the predicted change for the son is 0.75 inches.
* The regression line in the form: y = mx + b, is m (slope) = ρ * (σ<sub>y</sub>/σ<sub>x</sub>) and b (intercept) = μ<sub>y</sub> - (m * μ<sub>x</sub>). Using this, the regression line can be plotted in comparision to the whole dataset:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/regression-line-1.png" width = 200 height = 200>

* The regression line gives us the prediction for the conditional averages which is useful when there aren't enough values for certain stratas. Also, we use all the data just to estimate the slope and intercept which makes it much more stable because of a smaller standard error.
* The correlation and regression line can be misused or misinterpreted. These should be used when the dataset invloves a bivariate normal distribution, the scatterplot of these 2 variables should look like an oval. They can be thin (high correlation) or circle-shaped (no correlation):
<img src = "https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/bivariate_normal_distibutom.PNG" width = 300 height = 300>

* Bivariate normal distribution is defined for pairs (x and y with paired values). If x and y are both normally distributed random variables, and for any grouping/strata of x (x = largest_height_group) y is approximately normal in that group, then the pair is approximately bivariate normal. When x is set to a certain group its defined as conditional distribution of y given that x = largest_height_group.
* Galton says, when 2 variables follow a bivariate normal distribution then: E(y|x = any_strata_grouping) (for any given x the expected value of the y in pairs for which x is set at any_strata_grouping) = μ<sub>y</sub> + ρ * ((x - μ<sub>x</sub>)/σ<sub>x</sub>) * σ<sub>y</sub>. Notice, the slope and intercept for this line are the same as the regression line.
* So, if the data is approximatley bivariate, conditional expectation is given by the regression line. 
* The standard deviation of conditional distribution is: (y|x = any_strata_grouping) = σ<sub>y</sub> * sqrt(1 - ρ<sup>2</sup>). The variance of y is σ<sup>2</sup>, conditiong on x (basically, the variance of a conditional distribution) brings the variance down to: (1 - ρ<sup>2</sup>) * σ<sup>2</sup><sub>y</sub>. In the statement "X explains such and such percent of the variability," the percent value refers to the variance. The variance decreases by ρ<sup>2</sup> * 100%.
* The above, shows that correlation and the amount of variance explained are related. But, the variance explained statement only makes sense when the data is approximated by a bivariate normal distribution.
* When we predicted the son's height based on the father we used the equation for the regression line: E(y | x = X) = 35.7 + 0.5x (the E stands for expected value). What if we wanted to find the father's height based on the son's height, we would not find the inverse of the regression line. But, instead solve for the expected value of a conditonal (x | y = Y) of the father's height based on the son's height. This results in: E(x | y = Y) = 34 + 0.5y.
* Basically, the 2 regression lines of a dataset (x | y = Y and y | x = X) are not inverses of each other. It depends on which conditional expectation you compute for.

<strong>Linear Models:</strong>
* Association is not causation. For example, our baseball data shows that a team with 2 more base on balls per game than average scores 1.47 runs per game. But, this doesn't mean base on balls are the cause. The slope of the regression line for singles is 0.449 which is less than the slope for base on balls and a single gets you to first base just like base on balls. But, with a single runners that are on base have a higher chance of scoring than with base on balls.
* The reasons base on balls aren't predicitve of runs is because of confounding. Homerun hitters tend to have more base on balls since pitchers are more afraid to throw to them so actually, home runs cause the runs and not base on balls. We say bases on balls are confounded with home runs.
* Bases on balls can still be useful to create runs, we just need to factor in the home runs confounding. One approach is to keep home runs fixed at a certain value then examine the relationship between runs and base on balls, we can stratify home runs per game to the closest tenth. The regression slope for predicting runs with bases on balls (ignoring home runs) was 0.735. The slope of the stratified home run regression lines are much closer to the slope of singles, 0.449. This shows singles and base on balls have about the same effect on runs.
* Another approach is to stratify by base on balls and see if the home run effect goes down. The new slopes don't change that much from the original estimate of 1.84. This is consistent with the fact that base on balls do cause some runs. We have approximatley normal distributions for runs versus base on balls and runs versus home runs. 
* It's kind of complex to be computing regression lines for each strata so there might be an easier way. When random variability is taken into account, the estimated slopes for both strata don't seem to change by that much. If the slopes are the same than both conditionings (runs conditioned on base on balls & runs conditioned on home runs) are actually constant and can be written as (E = expected value, BB = base on balls, HR = home runs): E[R | BB = x<sub>1</sub>, HR = x<sub>2</sub>] = β<sub>0</sub> + β<sub>1</sub>x<sub>1</sub> + β<sub>2</sub>x<sub>2</sub>. This implies that if the number of home runs is fixed at x<sub>2</sub>, we observe a linear relationship between runs and base on balls with an intercept of β<sub>0</sub> + β<sub>2</sub>x<sub>2</sub>. The model also suggests that as the number of HR grows, the intercept growth is linear as well and determined by β<sub>1</sub>x<sub>1</sub>.
* The analysis above is reffered to as *multivariate regression* and it helps adjust for confoudning. This is the case because the base on balls slope (β<sub>1</sub>) has adjusted for the home run effect. But, we need to estimate for β<sub>1</sub> and β<sub>2</sub>.
* *Regression* allows us to find relationships between 2 variables while adjusting for others. This is really useful for confounding, like consider estimating the effect of eating fast foods on life expectancy using a random sample. Most fast food consumers are smokers, drinkers, and have lower incomes and a naive regression model may overesitmate the effects of fast food when not adjusting for confounding. 
* Adjusting for confounding can be done with regression. If the data is bivariate normal then the conditonal expectation follows a regression line. The conditional expectation following a regression line is not an assumption but a result derived from the assumption the data is approximatley bivariatie normal. In practice, its common to write down a model using 2 or more varaibles using a linear model. Note: linear doesn't refer to lines exclusivley but to the fact the condtional expectation is a linear combination of known quantaties.
* β<sub>0</sub> + β<sub>1</sub>x<sub>1</sub> + β<sub>2</sub>x<sub>2</sub> is a linear combination of x<sub>1</sub> and x<sub>2</sub>. The simplest linear model would be a constant, β<sub>0</sub>, next simplest would be a line, β<sub>0</sub> + β<sub>1</sub>x. 
* For the son and father heights dataset we can n observed father heights as: x<sub>1</sub>, ... , x<sub>n</sub>. And we can model the son's heights we're trying to predict as (x<sub>i</sub> = father's height, fixed not random due to the conditioning; y<sub>i</sub> = random son's height we want to predict; ϵ = errors, we assume they're independent of each other (expected value = 0, and standard deviation/sigma doesn't depend on i)): y<sub>i</sub> = β<sub>0</sub> + β<sub>1</sub>x<sub>i</sub> + ϵ<sub>i</sub>, i = 1, ... , N. 
* To have an useful model for prediction we need β<sub>0</sub> and β<sub>1</sub>, we have to estimate these from the data. Once, we find these values we can predict any son's height for father's height (x). If we assume the errors/epsilons are normally distributed then this is the same exact model we derived for the bivariate normal distribution. Linear models are just assumed without assuming normality. If the data is bivariate normal then the linear model holds, if it's not then you'll need other ways of justifying a linear model. Linear models are interpretable, in our case it can be interpreted as: due to inherited genes, the son's height prediction grows by β<sub>1</sub> for each inch we increase the father's height x. We need the term epsilon (ϵ) to include the remaining variability (like mother's genetic effect, enviromental factors, and other randomness). Least squares estimates can be ran in R using the lm() function: ```lm(son ~ father, data = galton_heights) #This creates a linear model for son's heights vs father's heights. Returning an intercept of 35.71 and coefficent of 0.5 for "father" which means for every inch we increase the father’s height, the predicted son’s height grows by 0.5 inches.```. The least squares estimate basically finds the best values for β<sub>0</sub> and β<sub>1</sub> for the linear model.
* The intercept (β<sub>0</sub>) isn't very interpretable as it's the predicted height of a son with a father who's 0 inches tall. We can rewrite the model to make the intercept parameter more interpretable (x̄ = average of x (average father height)): y<sub>i</sub> = β<sub>0</sub> + β<sub>1</sub>(x<sub>i</sub> - x̄) + ϵ<sub>i</sub>, i = 1, ... , N. This centers the covariate x<sub>i</sub>. Now β<sub>0</sub> is the predicted height for the son of the average father. We can run this in R by subtracting x̄ from each father's height value:
```r
galton_heights <- galton_heights %>%
    mutate(father_centered=father - mean(father))
lm(son ~ father_centered, data = galton_heights)
# This results in a new intercept of 70.45 and the "father_centered" coefficent is the same, at 0.5.
```
Least Squares Estimate:
* The betas (β<sub>0</sub> and β<sub>1</sub>) need to be estimated for linear models to be useful. So we need to find the coefficent values that have the least distance from the fitted model to the actual data. Residual sum of squares (RSS) measures the distance between the true value and the predicted value given by the regression line. The values that minimize the RSS are called the least squares estimates (LSE), the LSE values can be denoted with B̂<sub>0</sub> and B̂<sub>1</sub>. A function for RSS can be written as:
```r
rss <- function(beta0, beta1, data){
    resid <- galton_heights$son - (beta0+beta1*galton_heights$father)
    return(sum(resid^2))
}
```
* This creates a 3D plot with β<sub>0</sub> and β<sub>1</sub> as x and y axes and RSS as the z axis. Then, we can examine this graph to find the minimum or least squares estimate. Or we can just use the lm() function as I already mentioned: ```lm(predicting_variable ~ variables_using_to_predict, data = dataset)```. The function summary() can be used to extract all the data in the lm() once it runs (remember, the LSE are random variables).
* LSE are derived from random data (y<sub>1</sub>, ..., y<sub>n</sub>) which means the LSE are random variables. The t-distributions for the LSEs (shown in the summarize()) assume the epsilons follow a normal distribution. Using this, mathematical theory says that the LSE divide by their standard error (found in the summarize()) follow a t distribution with N - p degrees of freedom, p = number of parameters in the model (our case is 2, father and son). The 2p values are testing the null hypothesis (B̂<sub>0</sub> = 0 and B̂<sub>1</sub> = 0). For a large enough N, the CLT works and the t distribution becomes almost normal. So you can assume errors are normal and use the t distribution or assume N is a large enough for CLT to work and use normal distribution and construct confidence intervals for the parameters.
* Hypothesis testing for regression models is very common. This helps make statments like: the effect of A and B was statistically significant after adjusting for X, Y, and Z.
* LSEs can be strongly correlated and the correlation depends on how the predictors are defined or transformed (like standarizing the father's height which is changed from x<sub>i</sub> -> x<sub>i</sub> - x̄).
* We obtain predictions of y (son's height) by plugging in the estimates into our regression model. For example, father's height being x results in the son's height (ŷ) being: ŷ = B̂<sub>0</sub> + B̂<sub>1</sub> * x. Plotting ŷ versus x will give us the regression line. Also, ŷ is a random variable, mathematical theory tells us the standard errors. Assuming the errors are normal or have a large enough sample size allows us to use the CLT, which, in turn, allows us to create confidence intervals for our predictions. There is functon that allows us to create confidence intervals for the predicted ŷ: ```geom_smooth(method = "lm") #Just set a ggplot2 layer, geom_smooth()'s method to "lm"```:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/father-son-regression-1.png" width = 200 height = 200>

* If you just want the predictions the function, predict(), takes in an lm object as input and returns these predictions: ```predict(lm(son ~ father, data= galton_heights)```.

Dyplr:
* The lm() isn't part of the tidyverse so it doesn't know how to handle the outcome of a tidyverse function, like group_by, which is a tibble.
* The tibble is a special kind of data frame, the group_by() returns a special tibble, the grouped tibble. The functions, select, mutate, filter, and arrange return the class of the input, preserve the class of the input. Tibbles are the data frames for tidyverse, tibbles can be thought of as the modern version of data frames. <br>
Differences Between Tibbles and Data Frames:

    1. Tibbles display better, the print() for tibbles are much more readable than data frames.
    2. Subset of tibbles are tibbles. But, if you subset a data frame you might get back a different object, like an integer.
    3. Tibbles give you a warning if you try to access a column that doesn't exist. But, data frames don't give warnings which can make it hard to debug code.
    4. Columns of a data frame need to be a vector of numbers, string, or Booleans. But, tibbles can have columns with more complex objects like lists or functions.
    5. Tibbles can be grouped, group_by() returns a grouped tibble and this stores information that lets you know which rows are in which groups.
* Tidyverse functions return data frame objects to facilitate stringing via the pipe (%>%) operator. But, most R functions don't recognize tibbles nor do they return data frames (like the lm()).
* The do() function serves as a bridge between R functions, like lm(), and the tidyverse. It understands group tibbles and always returns a data frame. We need to include a column name so do() actually returns a data frame and doesn't return the actual output. For an useful data frame to be constructed the output of the function, inside do(), must be a data frame. But, don't name the data frame because then the name of the data frame will be the column name and the objects in the column will be data frame objects. And, if the data frame being returned has more than 1 row, they will be concatenated properly. An example:
```r
dat %>% 
  group_by(HR) %>% 
  do(get_slope(.))
```
* The broom package ([kind of cheatsheet](https://4va.github.io/biodatasci/handouts/r-stats-cheatsheet.pdf)) is designed to facilitate the use of model fitting functions (like lm()) with tidyverse. It has 3 main functions, which extract information from the object returned by the function lm() and return it in a tidyverse friendly data frame, called tidy(), glance(), and augment().
* The tidy() returns estimates and related information as a data frame, you can other important summaries (like confidence intervals) using arguments. Combining this with do() and lm() results in neat tables that make visualization with ggplot2 really easy:
```r
# use tidy to return lm estimates and related information as a data frame
library(broom)
fit <- lm(R ~ BB, data = dat)
tidy(fit)

# add confidence intervals with tidy
tidy(fit, conf.int = TRUE)

# pipeline with lm, do, tidy
dat %>%  
  group_by(HR) %>%
  do(tidy(lm(R ~ BB, data = .), conf.int = TRUE)) %>%
  filter(term == "BB") %>%
  select(HR, estimate, conf.low, conf.high)
  
# make ggplots
dat %>%  
  group_by(HR) %>%
  do(tidy(lm(R ~ BB, data = .), conf.int = TRUE)) %>%
  filter(term == "BB") %>%
  select(HR, estimate, conf.low, conf.high) %>%
  ggplot(aes(HR, y = estimate, ymin = conf.low, ymax = conf.high)) +
  geom_errorbar() +
  geom_point()
```
* The glance() function relates to model specific outcomes. It returns model specific summaries.
* The augment() function relates to observation specific outcomes It returns observation specific summaries.

Regression And Baseball:
* The linear model for runs per game is (y<sub>i</sub> = runs per game, x<sub>1</sub> = base on balls per game, x<sub>2</sub> = home runs per game): y<sub>i</sub> = β<sub>0</sub> + β<sub>1</sub>x<sub>1</sub> +  β<sub>2</sub>x<sub>2</sub> + ϵ<sub>i</sub>. To use lm() we tell it there are 2 predictor variables: 
```r
fit <- Teams %>% 
  filter(yearID %in% 1961:2001) %>% 
  mutate(BB = BB/G, HR = HR/G,  R = R/G) %>%  
  lm(R ~ BB + HR, data = .)
```
* The estimated slope with 1 variable were BB slope = 0.735 and HR slope = 1.844. But, with the multivariable model both effects/slopes go down with BB = 0.387 and HR = 1.561. To construct a metric to pick players we need to consider singles, doubles, and triples, too.
* We're going to assume these 5 variables are all normally distributed. So if we pick any 1 of them and hold the other 4 constant, the outcome (runs per game) is linear. And, the slopes for this relationship don't depend on the other 4 values that were held constant. If this model holds true then the linear equation is (x<sub>i, 1</sub> = BB, x<sub>i, 2</sub> = singles, x<sub>i, 3</sub> = doubles, x<sub>i, 4</sub> = triples, x<sub>i, 5</sub> = HR): y<sub>i</sub> = β<sub>0</sub> + β<sub>1</sub>x<sub>i, 1</sub> + β<sub>2</sub>x<sub>i, 2</sub> + β<sub>3</sub>x<sub>i, 3</sub> +  β<sub>4</sub>x<sub>i, 4</sub> + β<sub>5</sub>x<sub>i, 5</sub> + ϵ<sub>i</sub>. Using lm() we can find the LSE:
```r
fit <- Teams %>% 
  filter(yearID %in% 1961:2001) %>% 
  mutate(BB = BB / G, 
         singles = (H - X2B - X3B - HR) / G, 
         doubles = X2B / G, 
         triples = X3B / G, 
         HR = HR / G,
         R = R / G) %>%  
  lm(R ~ BB + singles + doubles + triples + HR, data = .)
```
* Now, to test this metric out we can use the predict() to predict number of runs for each team in 2002 since we didn't use the year 2002 in our sample data:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/model-predicts-runs-1.png" width = 200 height = 200>

* The metric does a pretty good job as the actual outcome points fall pretty close to the predicted values (which is the line).
* We can use our fitted model to form a more informative model that relates directly to run production. Specifically, to define a metric for player A we imagine a team of players just like A and use our fitted regression model to predict how many runs this team would score. Formula: -2.769 + (0.371 * BB) + (0.519 * singles) + (0.771 * doubles) + (1.240 * triples) + (1.433 * HR). We have derived a metric for teams based on team level summary statistics but for each indivual player we still need some more work to do.
* For players, a rate that takes into account oppurtunities is a per-plate-appearance. To make the per-game team rate comporable to the per-plate-appearance player rate we compute the average number of team plate appearances per game:
```r
# average number of team plate appearances per game
pa_per_game <- Batting %>% filter(yearID == 2002) %>% 
  group_by(teamID) %>%
  summarize(pa_per_game = sum(AB+BB)/max(G)) %>% 
  pull(pa_per_game) %>% 
  mean
```
* Now, we fit our model with player plate appearances and data from 1999 to 2001 to predict players in 2002. The player specific metrics are the number of runs we predict a team would score if all batters are like that player (if that player played the whole time). But, we can see wide variability:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/r-hat-hist-1.png" widht = 200 height = 200>

* We need to know the player's salary (limited budget) and player's position (fit team criteria). We need to do some data wrangling and combine data in different tables in the Lahman library. Once we take this into consideration and create a list of really good players that aren't too expensive then we've created our team!
* A way to pick players is via *[linear programming](https://brilliant.org/wiki/linear-programming/)*:
```r
library(reshape2)
library(lpSolve)

players <- players %>% filter(debut <= "1997-01-01" & debut > "1988-01-01")
constraint_matrix <- acast(players, POS ~ playerID, fun.aggregate = length)
npos <- nrow(constraint_matrix)
constraint_matrix <- rbind(constraint_matrix, salary = players$salary)
constraint_dir <- c(rep("==", npos), "<=")
constraint_limit <- c(rep(1, npos), 50*10^6)
lp_solution <- lp("max", players$R_hat,
                  constraint_matrix, constraint_dir, constraint_limit,
                  all.int = TRUE) 
```
* The on-base-percentage plus slugging percentage (OPS) is used by sabermetricians instead of batting average: BB/PA +(Singles+2Doubles+3Triples+4HR)/AB.
* Sophmore Slump is when a second effort fails to live up to the standard of the first effort, commonly used for apathy of students, performance of atheletes, singers/bands, and more. We can see if this slump exists in the baseball data by creating a table of player ID, names, and most played position. Also, we'll create a table of Rookie of the Year winners and add their batting statistics (batting averages) with players that played a sophmore season. Once this table is created, you can see that slump appears to be real as the batting average drops the 2nd year with a proportion of 68% batters having a lower batting average the 2nd year.
* Is this slump the jitters or a jinx? To figure this out we can look at all players and perform the similar operations we performed. Intrestingly, the top performers (who aren't rookies) go down the next year, even though its not their sophmore year. Also, the worst performers have their batting averages go up the next year. 
* Turns out, there's no such thing as a Sophmore Slump. Since the correlation for performance in 2 seperate years is high but not perfect. The 2013 and 2014 data for batting averages looks like a bivariate normal distribution:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/regression-fallacy-1.png" width = 200 height = 200>

* The correlation is 0.46 (not that strong) and if we're to predict the 2014 batting average (y) for a player that had a 2013 batting average (x) we would use the regression equation: (y - 0.225)/.032 = 0.46 * (x - .261/.023). Since the correlation is not prefect we except 2013 high performers to do a little worse in 2014 (regression to the mean) and the other way around. It's not a jinx.
* 2 or more variables, assuming the pairs are bivariate normal, which allow for a linear model cover most real life examples where linear regression is used. Another application is *measurment error models*.
* Measurment error models usually have nonrandom covariates (like time) and randomness is introduced through measurment eror other than sampling or natural variability.
* You and your scientific team are studying velocity and 1 person climbs the Tower of Pisa and drops a ball, other assistants record the position at different times. The falling_object dataset contains data on what that would look like and this is it graphed:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/gravity-1.png" width = 200 height = 200>

* By looking at the plot you deduce the position should follow a parabola which can be written as: f(x) =  β<sub>0</sub> + β<sub>1</sub>x + β<sub>2</sub>x<sup>2</sup>. The data doesn't exactly follow a parabola which is due to measurment error. To account for this we write (y<sub>i</sub> = distance in meters, x<sub>i</sub> = time in seconds, ϵ = measurment error): y<sub>i</sub> = β<sub>0</sub> + β<sub>1</sub><sub>x<sub>i</sub> + β<sub>2</sub><sub>i</sub>x<sup>2</sup> + ϵ<sub>i</sub>, i = 1, ... , N. Measurment error is assumed to be random, independent from each other, and having the same distribution from each eye. Also, it's assumed that measurment error has no bias so the expected value of epsilon is 0 (E[ϵ] = 0). This is a linear model (combination of known quantaties, x's, and unknown parameters, betas). Unlike the previous example the x's are fixed quantaties (time), no conditioning.
* To start actually predicting about other falling objects we need numbers so we need to solve for the unknown parameters. For this we should solve for the LSEs. The LSEs don't require the errors to be approximatley normal so we can use lm() to find the smallest RSS (LSE):
```r
fit <- falling_object %>%
    mutate(time_sq = time^2) %>%
    lm(observed_distance~time+time_sq, data=.)
tidy(fit)
```
* Now, we've recieved our estimated parameters and can use the broom function, augment(), to check if the predicted data fits the collected data:
<img src= = "https://rafalab.github.io/dsbook/book_files/figure-html/gravity-1.png" width = 200 height = 200>
    
* This fits with the formula for the velocity of a falling object and fits with the collected data.

<strong>Confounding:</strong>

Correlation Isn't Causation:
* We must be careful to not overinterpret the associations between variables using the tools and models we mentioned. There are many reasons x can correlate with y without either being a cause for the other.
* *Spurious correlations* is 1 way we can misinterpret associations. The correlation between divorce rates and margarine consumption is 0.93, which is really strong. This doesn't mean margarine consumption causes divorces or vice versa. Data dredging, data fishing, or data snooping is a form of cherry picking. Running a Monte Carlo of a large number of random variables and then finding their correlation, p-values, seeing a normal distribution would lead you to think these 2 variables have some statistical significance but they're just random. 
* With the Monte Carlo simulation, mentioned up above, the p-value is so small it makes it seems this is significant. This is called p-hacking and is a problem in scientifict publications (since publishers favor statistically significant results over negative ones). For example, researches may look for associations between average outcome and several exposures and only report the exposure with the smallest p-value. Also, they may create a lot of models for confounding and pick the one that has the smallest p-value.
* Another way to see high correlation when there's no causation are outliers. Let's say we take measurments for x and y and standarize the measurments but, we forgot to standarize entry 23. And, entry 23 might be a really big outlier and make the correlation as high as 0.99. Removing this outlier might even bring the correlation down to 0. One way to combat is to detect outliers and remove them. 
* The *Spearman correlation* helps estimate the population correaltion from the sample correlation which is robust to outliers. Basically, the correlation is computed on the ranks instead of the values (use rank() function). Because of this, the outlier doesn't drastically increase the correlation, almost to 0.99. Another way to calculate the Spearman correlation is via the cor() function: ```cor(x, y, method = "spearman")```
* One more way associations are confounded with causation are when cause and effect are reversed. One example of this is: tutoring makes students perform worse because they test lower than peers that aren't tutored, its the other way around. Also, they're probably getting tutored because they weren't performing well in the first place. Cause and effect reversal can be constructed with the Galton father and son height data (x<sub>i<sub> = father height data, y<sub>i<sub> = son height data): x<sub>i</sub> = β<sub>0</sub> β<sub>1</sub>y<sub>i</sub> + ϵ<sub>i</sub>, i = 1, ..., N. Even though we obtain correct results (p-values, estimates, standard of error, and etc.) but the mathematical interpretation of the model could suggest the son being tall caused the father to be tall, which isn't true.
* Confounders are probably the biggest reason that leads to associations to be misinterpreted. If x and y are correlated we call z a counfounder if changes in z cause changes in both x and y. If some criteria are met then we can use linear models to account for confounders. But, this is sometimes not possible, incorrect interpretations due to confounders happen a lot. 
* We can look at admissions data for UC Berkley in 1973, the total admissions show that 44% of men were addmitted compared to 30% women. But, the percent addmissions by major show that 4 out of the 6 majors favor women and all the differences are much smaller than the 14% difference (44% - 30%) we see when examing the total admissions. The totals show a dependence between admissions and gender but this disappears when grouping by gender. This can happen if an uncounted confounder is driving most of the variability. 
* To figure why this is we can define 3 variables, X: 1 for men and 0 for women, Y: 1 for admitted and 0 for not admitted, and Z for selectivity of major. A gender bias claim would be based on the fact that this Pr(Y = 1|X = x) is higher when X = 1 than when X = 0. Z is associated with Y since the higher the selecivity of that major the lower the probability someone is admitted into that major. But, is Z associated with gender (X)? We find out that there is some association, women were much more likely to apply to the 2 hardest majors (explaining the lower admission rates), gender and major selectivity being confounded. If we stratify by major, compute the differene, and then average we find that, in fact, women have a higher average by 3%, men = 38.2 and women = 41.7. This is an example of Simpson's paradox.
* *Simpson's paradox* is 
