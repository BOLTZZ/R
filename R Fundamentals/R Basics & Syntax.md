*Note: The code can be written into the R console in the RStudio. Or written as a script.*
# Objects:
```r
a <- 2 # Objects can be defined using <- or =.
b = 2
print(a)
print(b)
# Prints the values of the objects
ls()
# Prints outs the names of all the objects stored by the console.
```
# Functions:
```r
help("function_name")
# The above helps pull up the documentation of the function_name function.
```
Seq() is a pretty important function. It creates a sequence of numbers with a starting/maximum value and an increment amount (the default is 1).
```r
seq(starting_number, maximum_number, increment_amount) #The default increment_amount is 1.
x = seq(1, 100, 2)
print(x)
# This prints out a sequence of odd numbers from 1 to 100.
# There is another argument called length.out that changes the increment amount so the amount of numbers from the start number to the end number equal the value of lenght.out.
x = seq(0, 100, length.out = 5) #length.out = 5 so there will be 5 numbers from start to finish with the same increment.
print(x)
# This prints out: 0, 25, 50, 75, 100. As you can see there are 5 numbers with a constant increment amount of 25.
```
There a few important functions for manipulating vectors:
```r
x = c(1, 4, 3, 10, 5)
# The order function returns an index of what was the prev. positon of numbers and what they were moved to, once ordered.
index = order(x)
# For example, index = [1, 3, 2, 5, 4]. Compare this to x and you can see that the third number (3) of x was moved to the 2nd position 
# of index since its the 2nd smallest number.
# order can be useful to get the new position of vector elements.
# The sort function sorts the vector
sort(x)
# This results in x looking like this: x = [1, 3, 4, 5, 10]
# The rank function is similar to order but it matches up with the values in x.
ranked = rank(x)
#     x  = [1, 3, 4, 10, 5]
# ranked = [1, 2, 3, 5, 4]. As you can see, the ranked vector shows the position the items in x need to go to.
# NOTE: rank(x) orders the numbers from low to high but rank(-x) orders it from high to low.
# The min and max functions find the smallest and largest values in the vector.
smallest = min(x)
# smallest = 1
largest = max(x)
# largest = 10
# The which.min and which.max functions locate the index of the smalles and largets values in the vector.
smallest_index = which.min(x)
# smallest_index = 1
largest_index = which.max(x)
# largest_index = 4
```
Locating certain indexes in vectors:
```r
# The function which() gives the entries of a logical vector that are true:
index_of_oregon = which(murders$state == "Oregon")
Syntax:
index = which(logical_vector)
# The function match() looks for similar entries in two vectors and returns the indexes on where to find them;
ny_and_ca = match(c("New York", "California"), murders$state)
Syntax:
index = match(vector_1, vector_2)
# %in% function to see if each element of the first vector is in the second vector.
a = c(1, 5, 4)
b = c(1, 10, 12)
a%in%b
```
Ploting data is very important for data visualization (the functions below are only a small fraction on how to plot data):
```r
# a simple scatterplot of total murders versus population
x <- murders$population /10^6
y <- murders$total
plot(x, y)
# a histogram of murder rates
hist(murders$rate)
# boxplots of murder rates by region
boxplot(rate~region, data = murders)
# Another box plot:
library(dplyr)
library(ggplot2)
library(dslabs)
data("murders")
murders %>% mutate(rate = total/population*100000, region = reorder(region, rate, FUN = median)) %>% group_by(region) %>% ggplot(aes(region, rate)) + geom_boxplot() + geom_point()
```
Output:

![1](https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/graphs.PNG)

If/else conditional in 1 line:
```r
ifelse(boolean_condition, return_this_if_true, return_this_if_true)
x = 5
y = 3
ifelse(x > y, print("x is greater than y"), print("y is greater than x"))
# This ifelse line is really important because it works on vectors, element for element.
```
Any and all functions:
```r
# The any() function takes in logicals and returns TRUE if any of the elements are TRUE:
logical_vector = c("FALSE", "FALSE", "TRUE", "FALSE")
any(logical_vector)
# any() returns TRUE because of that one TRUE.
# The all() function takes in logicals and returns TRUE only if all the elements are TRUE:
all(logical_vector)
# all() returns FALSE because of the 3 FALSE elements.
```
Creating and defining function syntax:
```r
function_name = function(input, another_input_if_u_want){
  # Code:
  input_10 = input * 10
  # Note: R returns the last statement in the function unless there is a return statement:
  return(input_10)
}
```
# Datatypes:
```r
a = 2
class(a) 
# class(object) prints out the datatype of the object.
```
Data frames:
* Data frames are used to store datasets. They can combine a whole plethora of data into 1 object.
* Data frames can be though of as tables, rows represent observations and columns represent variables.
* The str(object) function is really useful for data.frame objects. It stands for structure of an object and gives a lot of in depth info. about a certain object.
* The head(object) function helps get information about the first 6 rows of a data.frame object.
* The dollar sign ($) is an accessor that gives all the data points of a certain variable (column). Also, square brackets can be used to do the same thing (data.frame object[["variable"]]).
This can be done by:
```r
Syntax:
data.frame object$variabe
Ex:
murders$population
# Note: The returned values of the population column of the murders data.frame object (table) preserve the order of the rows.
# To figure out what variables (columns) a certain data.frame object has class str(data.frame object). Or names(data.frame object) can be used to get the names of the columns of a data.frame object.
str(murders)
names(murders)
# Also, data.frame objects can be created. 
Syntax:
data.frame_object_name = data.frame(column_name = vector, another_column = another_vector, and so on..)
```
# Vectors:
* Vectors are objects with many values. Usually, the dollar sign accessor returns a vector because that variable which is being accessed most likely has many data points. 
```r
pop = murders$population
# pop would be a vector since it has many rows of population.
length(pop)
# Calling the length of an object returns how many values it has. Since, pop is a vector it will return a number larger than 1. Though, technically, vectors with length 1 are still considered to be vectors.
```
* Vectors have datatypes so there are numeric vectors (numbers), character vectors (""), logical vectors (true/false(boolean)), factor vectors, integer vectors (they can be made by adding a L to the end), and a few mor.
* Integer vectors and numerical vectors can be used interchangeably and in arithmetic operatons. The only difference is integers take up less computer memory than numerics which can come to be important in large computations.
```r
integer_vector = c(3L, 4L, 5L)
# An integer vector.
difference = 10L - 3
# The value of difference is 7.
```
* Factor vectors store categorial data like regions of the U.S. (Northeast, South, and etc).
```r
levels(murders$region)
# The above prints out the levels in the factor vector of region. This prints out 4 leveles (Northeast, South, West, and North Central).
```
* Factor vectors can be a source of confusion because they can look like character vectors. Only calling the class can help decipher if its a factor or character vector.
* Vectors can be created with c which stands for concatenate. 
```r
people = c("David", "Jacob", "Reyes")
# Creates a character vector up above.
ages = c(38, 15, 60)
# Creates a second numeric vector up above.
names(ages) <- people
# The names() function helps assign character names to each item of a vector.
# In this case, each age is assinged a person name.
# Output:
# David  Jacob  Reyes
# 38      15     60
```
* Square brackets [] help access elements of a vector (kinda like accessing array elements in Java and list elements in Python). This is called subsetting.
```r
ages[2]
# Output:
# Jacob
# 15
ages[1:3]# This returns all the elements from 1 to 3.
# Output:
# David  Jacob  Reyes
#  38     15     60
ages[c(1, 3)] # This returns 1st and 3rd element.
# Output:
# David  Reyes
#  38     60
# Subsetting can, also, be done for characters/strings.
ages["David"]
# Output
# David
# 38
```
* Vector coercien is when R tries to guess the dataypes.
```r
ages <- c(38, "Jacob", 60)
# Up above, R coerces the dataype of ages vector to character instead of numeric because "Jacob" is in the vector. And, an error is not thrown.
```
# Casting:
* as.character() turns a different dataype into characters.
* as.numeric() turns a different dataype into numbers.
```r
x <- 1:5
y <- as.character(x)
# Casts x into a character datatype.
```
# Logical Operators:
<table>
<colgroup span = "4"></colgroup>
<thead>
  <tr>
     <th colspan = "4" scope = "colgroup">Logical Operator Comparision</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Operator:</td>
    <td><a href = "https://github.com/BOLTZZ/Python">Python:</a></td>
    <td><a href = "https://github.com/BOLTZZ/Java">Java:</a></td>
    <td><a href = "https://github.com/BOLTZZ/R">R:</a></td>
  </tr>
  <tr>
    <td>NOT</td>
    <td>not</td>
    <td>!</td>
    <td>!</td>
  </tr>
  <tr>
    <td>OR</td>
    <td>or</td>
    <td>||</td>
    <td>|</td>
  </tr>
  <tr>
    <td>AND</td>
    <td>and</td>
    <td>&&</td>
    <td>&</td>
  </tr>
</tbody>
</table>
Another operator is the pipe operator (%>%):

```r
# The pipe operator can help write less code
my_states = filter(murders, region %in% c("Northeast", "West") & rate < 1) 
select(my_states, state, rate, rank)
# For example, up above I need 2 lines to create a data.frame object that only contains states that are in the Northeast or West and 
# have a murder rate less than 1. Then, using the select() function I only print out the state name, murder rate, and rank.
# But, down below I do the same exact thing only in 1 line because of the pipe operator:
filter(murders, region %in% c("Northeast", "West") & rate < 1) %>% select(state, rate, rank)
# The pipe operator takes in the previous function's return value/object as the inputted value/object, allowing this to work.
```
# Import Data Gathering Functions:
```r
data(heights)
x = heights$height
mean(x) # Finds the mean.
median(x) # Finds the median.
sd(x) # Finds the standard deviation (SD).
mad(x) # Finds the median absolute deviation (MAD).
# MAD is a robust summary because it's much more resistant to errors inputted by the person entering the data than SD or mean.
sample(vector, num_of_items_to_pick) # Finds a random item(s), based on the value of num_of_items_to_pick, from the vector.
replicate(num_of_times_to_be_repeated, function_to_be_repeated) # This can emulate a Monte Carlo simulation. Set num_of_times_to_be_repeated to a very high number (10,000). Also, function_to_be_repeated is usually sample().
paste(vector_1, vector_2) # Paste concatenates vectors or characters together.
expand.grid(pants = c("blue", "black"), shirts = c("white", "grey", "plaid")) # Gives combinations for 2 vectors.
permutations(n, r, dataset) # lists the different ways that r items can be selected from a set of n options when order matters.
combinations(n, r, datset) # lists the different ways that r items can be selected from a set of n options when order does not matter.
sapply(vector, operation_or_function) # sapply allows element-wise operations/functions on vectors.
# The quantile for normal distribution in R is qnorm():
qnorm(quantile, avg, sd) # returns the data at that quantile.
# Example (male heights at 99th percentile):
99th_percentile = qnorm(0.99, mean(x), sd(x))
# To take a sample of n numbers:
take_poll(n)
# Save objects into an rda file, .RData is also fine but less preferred:
save(murders, file = "rda/murders.rda")
# Save plots:
ggsave("figs/barplot.png")
# Map (similar to Python map()) function in the purr package:
map(list_or_iterable, function)
# Also, map() can be used to extract the ith entry of each element x:
map(i, x)
# Note: map() always returns a list, use map_chr() for character and map_int() for integer.
# The recode() can change long factor names (throughout the entire dataset) with the tidyverse package:
dataset %>% recode(variable_or_column_name, 'long_factor_name_1' = 'shorter_factor_name_1', 'long_factor_name_2' = 'shorter_factor_name_2')
# The cor() finds the correlation coefficent of 2 variables:
cor(variable_1, variable_2)
# The lm() finds the least squares estimate, useful for creating linear models:
lm(predicting_variable ~ variables_using_to_predict, data = dataset)
#  The do() function serves as a bridge between R functions, like lm(), and the tidyverse.
do(lm(R ~ BB, data = Teams))
# The createDataPartition() randomly splits data.
createDataParition(y_value_outcome_value_, times = 1, p = 0.5, list = FALSE) # arg times = how many random samples of indexes to return; arg p = proportion of index represented; arg list = indexes returnes as a list or not.
# Finds the confusion matrix:
confusionMatrix(data = y_hat, reference = test_set$sex) # The function expects factors as inputs and the 1st level is considered the positive outcome or y = 1. In this case, female is positive since it comes before male alphabetically.
# Finds the F1 value:
F_meas(data = y_hat, reference = factor(train_set$sex)) #Since beta defaults to 1 we don't need to change the value of beta.
# Finds the specificity of a prediction:
specificity(predicted_outcome_data, actual_data)
# Finds the sensitivity of a predciction:
sensitivity(predicted_outcome_data, actual_data)
```
