*Note: Check out [data science functions](https://github.com/BOLTZZ/R/blob/master/R%20Fundamentals/R%20Basics%20&%20Syntax.md#import-data-gathering-functions) for data science related functions.*
# Linear Regression:
* Linear regression is commonly used to quantify the relationship between two or more variables. It is also used to adjust for confounding. 

<strong>Regression Overview:</strong>
* Bill James was the originator of the sabermetrics, the approach of using data to predict what outcomes best predicted if a team would win.
* There is chance involved in scoring runs in baseball because sometimes hitting it too hard can make it catchable if a catcher is in the right position. This chance makes it good for data analysis.
* The batting averages (hits/# of times at bat) is considered one of the most important offensive statistic. But, this ignores bases on balls which is a success rate.
* Do teams with more home runs score more runs? We can use the Lahman library and graph this as a scatterplot because of the 2 variables. There is a strong, positive correlation showing that teams with more home runs score more runs. We can graph bases on balls and runs and we, also, see a pretty strong, positive correlation. Does this mean bases on balls cause more runs? Probably not, it appears bases on balls are causing runs but home runs are causing both (bases on balls and runs). This is called confounding.

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

* The regression line for 2 variables, x and y, says that for every standard deviation (σ<sub>x</sub>) increase above the average (μ<sub>x</sub>), y grows ρ (rho) standard deviations (σ<sub>y</sub>) above the average (μ<sub>y</sub>). Formula: ((y<sub>i</sub> - μ<sub>y</sub>)/σ<sub>y</sub>) = ρ * ((x<sub>i</sub> - μ<sub>x</sub>)/σ<sub>x</sub>). 
* For perfect correlation we predict increase that is the same number of sd. If there is no correlation we don't use x at all for the prediction of y. Values between 0 and 1, prediciton is somewhere in between. Negative correlation then predict a reduction instead of increase.
* When the correlation is positive but smaller than 1 that we predict something closer to the mean is called regression. The son *regresses* to the average height.
* The regression line in the form: y = mx + b, is m (slope) = ρ * (σ<sub>y</sub>/σ<sub>x</sub>) and b (intercept) = μ<sub>y</sub> - (m * μ<sub>y</sub>). Using this, the regression line can be plotted in comparision to the whole dataset:
<img src = "https://rafalab.github.io/dsbook/book_files/figure-html/regression-line-1.png" width = 200 height = 200>

* The regression line gives us the prediction for the conditional averages which is useful when there aren't enough values for certain stratas. Also, we use all the data just to estimate the slope and intercept which makes it much more stable because of a smaller standard error.
* The correlation and regression line can be misused or misinterpreted. These should be used when the dataset invloves a bivariate normal distribution, the scatterplot of these 2 variables should look like an oval. They can be thin (high correlation) or circle-shaped (no correlation):
<img src = "https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/bivariate_normal_distibutom.PNG" width = 300 height = 300>

* Bivariate normal distribution is defined for pairs (x and y with paired values). If x and y are both normally distributed random variables, and for any grouping/strata of x (x = largest_height_group) y is approximately normal in that group, then the pair is approximately bivariate normal. When x is set to a certain group its defined as conditional distribution of y given that x = largest_height_group.
* Galton says, when 2 variables follow a bivariate normal distribution then: E(y|x = any_strata_grouping) (for any given x the expected value of the y in pairs for which x is set at any_strata_grouping) = μ<sub>y</sub> + ρ * ((x - μ<sub>x</sub>)/σ<sub>x</sub>) * σ<sub>y</sub>. Notice, the slope and intercept for this line are the same as the regression line.
* So, if the data is approximatley bivariate, conditional expectation is given by the regression line. 
* The standard deviation of conditional distribution is: (y|x = any_strata_grouping) = σ<sub>y</sub> * sqrt(1 - ρ<sup>2</sup>). The variance of y is σ<sup>2</sup>, conditiong on x (basically, the variance of a conditional distribution) brings the variance down to: (1 - ρ<sup>2</sup>) * σ<sup>2</sup><sub>y</sub>. In the statement "X explains such and such percent of the variability," the percent value refers to the variance. The variance decreases by ρ<sup>2</sup>%.
* The above, shows that correlation and the amount of variance explained are related. But, the variance explained statement only makes sense when the data is approximated by a bivariate normal distribution.
* When we predicted the son's height based on the father we used the equation for the regression line: E(y | x = X) = 35.7 + 0.5x (the E stands for expected value). What if we wanted to find the father's height based on the son's height, we would not find the inverse of the regression line. But, instead solve for the expected value of a conditonal (x | y = Y) of the father's height based on the son's height. This results in: E(x | y = Y) = 34 + 0.5y.
* Basically, the 2 regression lines of a dataset (x | y = Y and y | x = X) are not inverses of each other. It depends on which conditional expectation you compute for.