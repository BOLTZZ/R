*Note: Check out [data science functions](https://github.com/BOLTZZ/R/blob/master/R%20Fundamentals/R%20Basics%20&%20Syntax.md#import-data-gathering-functions) for data science related functions.*
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
# Getting min, median, and max using quantiles:
quantile(data, c(0, 0.5, 1))
# All percentiles of the set below:
p = seq(0.01, 0.99, 0.01)
# The above gets all the percentiles because of seq()'s syntax = seq(first_value, last_value, increment)
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
* Sometimes normal distribution doesn't fit the data so we can't give a 2 number summary of the mean/median ± standard deviation using a QQ plot or a smooth density plot. So we can give a 5 number summary (max., min., L.Q., U.Q., and median) using a boxplot. Or, we can just use a histogram.
* Using explatory data analysis (plots/graphs) we can find errors by finding unusual outliers.
Few Components of a Graph:
1. Data - What data the graph contains.
2. Geometry - What type of graph is it? Scatterplot, histogram, QQ plot, bar graph?
3. Aesthetic mappings - What are both axes (x and y) used to represent? What is text used for? If the graph contains a legend or differentiated colors, what are they used to represent?
4. Scale - Is the data scaled down, if so, by what factor (log, 2, etc)? Does the data all fit on both axes?
5. Labels, style, title, legend, and more.
For a good example of a graph click [here](https://github.com/BOLTZZ/R/blob/master/R%20Projects/Murders%20Graph.md)
The gridExtra pacakage can be used to arrange a lot of graphs next to each other:
```r
library(ggplot2)
# Note ggplot2 package makes it so plots can be saved as packages.
# define different plots
p <- heights %>% filter(sex == "Male") %>% ggplot(aes(x = height))
p1 <- p + geom_histogram(binwidth = 1, fill = "blue", col = "black")
p2 <- p + geom_histogram(binwidth = 2, fill = "blue", col = "black")
p3 <- p + geom_histogram(binwidth = 3, fill = "blue", col = "black")
# arrange plots next to each other in 1 row, 3 columns
library(gridExtra)
grid.arrange(p1, p2, p3, ncol = 3)
```
dplyr/tidyvers functions:
```r
# The summarize function creates a data frame with inputted columns:
s = heights %>% filter(sex == "Male") %>% summarize(average = mean(height), standard_deviation = sd(height))
# Access average and standard deviation from summary table:
s$average
s$standard_deviation
# The dot and %>% can be used to get a certain column of a data frame:
data_frame %>% .column1
rate = us_murder_rate %>% .$rate
# The group_by() helps divide data into groups them compute summaries for each group.
heights %>%
    group_by(sex) %>%
    summarize(average = mean(height), standard_deviation = sd(height))
# This summarizes the mean and sd for each sex.
# The arrrange() can sort data depending on a certain value:
murders %>% arrange(population)
# The code above sorts the murders dataset based on population (lowest to highest).
murders %>% arrange(desc(population))
# This sorts it from highest to lowest.
# arrange() can, also, be nested:
murders %>% arrange(region, murder_rate)
# This orders it by region then by murder rate.
# To see the top something of a dataset use the top_n() function.
dataset $>$ top_n(max_number, column)
murders %>% top_n(10, murder_rate)
# Finds the top 10 states based on murder rate (this is UNORDERED).
murders %>% arrange(desc(murder_rate)) %>% top_n(10)
# The above orders it using arrange().
# Another important function is na.rm(). na.rm() removes any NAs and is useful when operating on datasets with NAs
average = mean(murders$rate, na.rm = TRUE)
# Removes any possible NAs from the calculations.
```
Different Ways to Compare Data:
* The geom_density() can be used to create a smooth density plot.

Faceting:

Facting - Stratifying the data by some variable and then making a plot for each variable.

The facet_grid() function takes in variable(s) and adds a new layer whilst plotting the strata(s).
```r
filter(gapminder, year %in% c(1962, 2012)) %>%
    ggplot(aes(fertility, life_expectancy, col = continent)) +
    geom_point() +
    facet_grid(continent ~ year)
# The facet_grid() takes in 2 variables (continent and year) seperated by a ~
facet_grid(var ~.)
# The above puts the graphs vertically next to each other.
facet_grid(.~var)
# The above puts the graphs horizontally next to each other.
# The facet_wrap() function allows you to seperate the plots in either columns or rows.
# Keeping the scales (x and y axis) the same between facets helps compare.
```
Time-Series Plots:

Time-series plots - Have time on the x axis and series on the y-axis.

* geom_line() can be used to connect points and create a smooth line.

* When 2 or more variables exist on the graph and geom_line() is called just use, "group = variable" in the mapping:
```r
countries <- c("South Korea", "Germany")
gapminder %>% filter(country %in% countries) %>%
    ggplot(aes(year, fertility, col = country)) +
    geom_line()
# Each line is colored differently
labels <- data.frame(country = countries, x = c(1975, 1965), y = c(60, 72))
gapminder %>% filter(country %in% countries) %>%
    ggplot(aes(year, life_expectancy, col = country)) +
    geom_line() +
    geom_text(data = labels, aes(x, y, label = country), size = 5) +
    theme(legend.position = "none")
# The above is a life expectancy plot without a legend but instead titles over each line.
```
Transformations:
* Log transformations help convert mulitplicative changes into additive changes.
* Most common logs are log2, log10, and the natural log (though this is hard to compute for humans).
* You can divide an axis (usally the x-axis) by logN to lower the numbers lined on the x-axis and make them smaller.
* For population size, log10 seems the best because population size numbers are, generally, very large.
* Choosing binwidths for histograms and smooth density plots becomes challenging as we convert the numbers based on logN.
* Scaling based on logN:
```r
scale_x_continous(trans = "logN")
scale_y_continous(trans = "logN")
```
Boxplots:

* When there are many boxplots in the same graph the x-axis titles can get messed up so use hjust = 1 so the names on the x-axis can be rotated:
```r
p + geom_boxplot() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
```
* We can use the reorder() function to reorder box plots based on meaningful values:
```r
mutate(first_vector = reorder(first_vector, second_vector, FUN = summarization_function))
# Mutate helps reorder
```
Logistic (logit) transformation for a proportion or rate p is as follows: f(p) = log(p/1-p). When p is a proportion or probability (p/1-p) is called the odds. The logit hilghits small differences like 0.1% using the odds and is useful for that.
