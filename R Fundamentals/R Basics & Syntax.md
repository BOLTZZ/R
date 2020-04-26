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
```
# Vectors:
* Vectors are objects with many values. Usually, the dollar sign accessor returns a vector because that variable which is being accessed most likely has many data points. 
```r
pop = murders$population
# pop would be a vector since it has many rows of population.
length(pop)
# Calling the length of an object returns how many values it has. Since, pop is a vector it will return a number larger than 1. Though, technically, vectors with length 1 are still considered to be vectors.
```
* Vectors have datatypes so there are numeric vectors (numbers), character vectors (""), logical vectors (true/false(boolean)), factor vectors, and more.
* Factor vectors store categorial data like regions of the U.S. (Northeast, South, and etc).
```r
levels(murders$region)
# The above prints out the levels in the factor vector of region. This prints out 4 leveles (Northeast, South, West, and North Central).
```
* Factor vectors can be a source of confusion because they can look like character vectors. Only calling the class can help decipher if its a factor or character vector.
