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
# The above helps pull up the documentation of the funciont_name function.
```
* Seq() is a pretty important function. It creates a sequence of numbers with a starting/maximum value and an increment amount (the default is 1).
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
* There a few important functions for manipulating vectors:
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
