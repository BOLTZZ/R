# Data Wrangling:
[Data Wrangling Cheatsheet](https://rstudio.com/wp-content/uploads/2015/02/data-wrangling-cheatsheet.pdf)

<strong>Data Import:</strong>
* One of the most common ways to store and share data is through spreadsheets (file version of a data frame).
* Spreadsheets have rows seperated by returns and columns separated by a delimiter. The most common delimiters are comma, semicolon, white space and tab.
* Many spreadsheets are raw text files and can be read with any basic text editor. However, some formats are proprietary and cannot be read with a text editor, such as Microsoft Excel files (.xls). Most import functions assume that the first row of a spreadsheet file is a header with column names. To know if the file has a header, it helps to look at the file with a text editor before trying to import it.
* The working directory is where R looks for files and saves files by default. On RStudio navigate to Session then Set Working Directory to set a working directory (it's suggested to create a new directory for each new project and keep the raw data in that directory).
* readr is a library of the tidyverse package that contains functions for reading data stored in text file spreadsheets into R ([cheatsheet](https://rawgit.com/rstudio/cheatsheets/master/data-import.pdf)):

| Function | Format | Typical Suffix
| -- | -- | --- 
| read_table | white space seperated values | txt 
| read_csv | comma separated values | csv
| read_csv2 | semicolon separated values | csv 
| read_tsv | tab delimited separated valaues | tsv
| read_delim | general text file format, must define delimeter | txt
* readxl is a library of the tidyverse package that contains functions for reading data stored in Microsoft Excel format:

| Function | Format | Typical Suffix
| -- | -- | --- 
| read_excel | auto detect the format | xls, xlsx 
| read_xls | original format | xls
| read_xlsx | new format | xlsx 
* The excel_sheets() function gives the names of the sheets in the Excel file. These names are passed to the sheet argument for the readxl functions read_excel(), read_xls() and read_xlsx().
* The read_lines() shows the first few lines of a file in R, this can be used to make sure your using the correct functions in readr and readxl. Also, readr and readxl functions result in objects of class tibble.
* The read_csv() and other import functions can read files on the web by using the url of the data. If you want to have a local copy of the file, you can use download.file(). To help you out use tempdir() and tempfile(). tempdir() creates a directory with a name that is very unlikely not to be unique. tempfile() creates a character string that is likely to be a unique filename. Then, just delete the files and directories:
```r
url <- "https://raw.githubusercontent.com/rafalab/dslabs/master/inst/extdata/murders.csv"
dat <- read_csv(url)
download.file(url, "murders.csv")
tempfile()
tmp_filename <- tempfile()
download.file(url, tmp_filename)
dat <- read_csv(tmp_filename)
file.remove(tmp_filename)
```
* Also, if your file doesn't have any headers you can use the col_names = False:
```r
read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", col_names = FALSE)
```
* R-base import functions (read.csv(), read.table(), read.delim()) generate data frames rather than tibbles and character variables are converted to factors. This can be avoided by setting the argument stringsAsFactors=FALSE.

<strong>Tidy Data:</strong>

Reshaping Data:
* In tidy data, each row represents an observation and each column represents a different variable.
* In wide data, each row includes several observations and one of the variables is stored in the header:
* Once you import data you need to reshape it for data analyzation. To reshape we use the tidyr package (in tidyverse) functions. One function is gather(), it turns wide data into tidy data. Note: the column names produced by gather are characters, this can be combated with the conver argument and setting it to TRUE. Another function is spread() and it converts tidy data to wide data. spread takes 3 arguments: (1) the data frame, (2) the key to spread across columns, and (3) the value to put in individual cells of the table:

![1](https://rafalab.github.io/dsbook/wrangling/img/gather-spread.png)

(Image courtesy of RStudio. CC-BY-4.0 license. Cropped from original.)

Code:
```r
# original wide data
library(tidyverse) 
path <- system.file("extdata", package="dslabs")
filename <- file.path(path,  "fertility-two-countries-example.csv")
wide_data <- read_csv(filename)

# tidy data from dslabs
library(dslabs)
data("gapminder")
tidy_data <- gapminder %>% 
  filter(country %in% c("South Korea", "Germany")) %>%
  select(country, year, fertility)

# gather wide data to make new tidy data
new_tidy_data <- wide_data %>%
  gather(year, fertility, `1960`:`2015`) #1st arg = sets the name of the column that will hold the variable, currently kept in the wide data; 2nd arg = sets the column name for the column that will hold the values in the column cells; 3rd arg = columns that will be gathered.
head(new_tidy_data)

# gather all columns except country
new_tidy_data <- wide_data %>%
  gather(year, fertility, -country)

# gather treats column names as characters by default
class(tidy_data$year)
class(new_tidy_data$year)

# convert gathered column names to numerictho
new_tidy_data <- wide_data %>%
  gather(year, fertility, -country, convert = TRUE)
class(new_tidy_data$year)

# ggplot works on new tidy data
new_tidy_data %>%
  ggplot(aes(year, fertility, color = country)) +
  geom_point()

# spread tidy data to generate wide data
new_wide_data <- new_tidy_data %>% spread(year, fertility) #1st arg = which variable for the column names; 2nd arg = which variable to fill out the cells.
select(new_wide_data, country, `1960`:`1967`)
```
* The separate() function splits one column into two or more columns at a specified character that separates the variables. When there is an extra separation in some of the entries, use fill="right" to pad missing values with NAs, or use extra="merge" to keep extra elements together. The unite() function combines two columns and adds a separating character. unite takes 3 arguments: (1) the data frame, (2) the name of the new column to create, and (3) a vector of the columns to unite with an underscore, in order:
```r
# separate then spread
dat %>% separate(key, c("year", "variable_name"), sep = "_", extra = "merge") %>%
  spread(variable_name, value) 
# full code for tidying data
dat %>% 
  separate(key, c("year", "first_variable_name", "second_variable_name"), fill = "right") %>%
  unite(variable_name, first_variable_name, second_variable_name, sep="_") %>%
  spread(variable_name, value) %>%
  rename(fertility = fertility_NA)
```
Combining Tables:
* The join functions in the dplyr package combine two tables such that matching rows are together, they only work for dataframes. In left_join() all rows in the left-hand table are retained and columns from both tables are added together. The right_join() retains rows from the right-hand table and columns from both tables are added together. The inner_join() only keeps rows that have information in both tables. The full_join() keeps all rows from both tables. The semi_join() keeps the part of first table for which we have information in the second. The anti_join() keeps the elements of the first table for which there is no information in the second.

<img src = "https://rafalab.github.io/dsbook/wrangling/img/joins.png" width = 300 height = 350>

(Image courtesy of RStudio. CC-BY-4.0 license. Cropped from original.)
* Unlike the join functions, the binding functions do not try to match by a variable, but rather just combine datasets (they need must match on the appropriate dimension, either same row or column numbers). bind_cols() binds two objects by making them columns in a tibble. The R-base function cbind() binds columns but makes a data frame or matrix instead. The bind_rows() function is similar but binds rows instead of columns. The R-base function rbind() binds rows but makes a data frame or matrix instead.
* The set operators in base R works on vectors but if tidyverse/dplyr are loaded, they also work on data frames. You can take intersections of vectors using intersect(). This returns the elements common to both sets. You can take the union of vectors using union(). This returns the elements that are in either set.
* The set difference between a first and second argument can be obtained with setdiff(). Note that this function is not symmetric, switching arguments can result in different answers. The function set_equal() tells us if two sets are the same, regardless of the order of elements:
```r
# intersect vectors or data frames
intersect(1:10, 6:15)
intersect(c("a","b","c"), c("b","c","d"))
tab1 <- tab[1:5,]
tab2 <- tab[3:7,]
intersect(tab1, tab2)

# perform a union of vectors or data frames
union(1:10, 6:15)
union(c("a","b","c"), c("b","c","d"))
tab1 <- tab[1:5,]
tab2 <- tab[3:7,]
union(tab1, tab2)

# set difference of vectors or data frames
setdiff(1:10, 6:15)
setdiff(6:15, 1:10)
tab1 <- tab[1:5,]
tab2 <- tab[3:7,]
setdiff(tab1, tab2)

# setequal determines whether sets have the same elements, regardless of order
setequal(1:5, 1:6)
setequal(1:5, 5:1)
setequal(tab1, tab2)
```
Web Scraping:
* Web scraping/web harvesting is extracting data from a website (when a data file doesn't exist). The information used by a browser to display data is recieved from a server as text, this text is written in HTML. Viewing the code and installing the HTML file of a webpage and bringing it into R can help get your data.
* The rvest is a web harvesting package of tidyverse. It can proccess HTML and XML (general markup language, HTML is a specific type of XML) files. rvest has functions to extract nodes (<>) from an HTML document, html_nodes() extracts all nodes of different types, and html_node() extracts the first node. To extract a table, use: html_doct %>% html_nodes("table"). Then, use html_table() to convert a HTML table to a data frame:
```r
# import a webpage into R
library(rvest)
url <- "https://en.wikipedia.org/wiki/Murder_in_the_United_States_by_state"
h <- read_html(url)
class(h)
h

tab <- h %>% html_nodes("table")
tab <- tab[[2]]

tab <- tab %>% html_table
class(tab)

tab <- tab %>% setNames(c("state", "population", "total", "murders", "gun_murders", "gun_ownership", "total_rate", "murder_rate", "gun_murder_rate"))
head(tab)
```
* The html_nodes() and html_node() use CSS selectors, but these can start to get complex. So, to easily find the selectors we can use a Chrome extension, [SelectorGadget](https://selectorgadget.com/). Using, this we can find the CSS selectors and extract the wanted data by using those selectrs and the html_nodes() and html_node() functions:
```r
h <- read_html("http://www.foodnetwork.com/recipes/alton-brown/guacamole-recipe-1940609")
recipe <- h %>% html_node(".o-AssetTitle__a-HeadlineText") %>% html_text()
prep_time <- h %>% html_node(".m-RecipeInfo__a-Description--Total") %>% html_text()
ingredients <- h %>% html_nodes(".o-Ingredients__a-Ingredient") %>% html_text()
# This can be turned into a function because recipe pages from websites follow the same, general layout.
```
<strong>String Processing:</strong>

* Common Tasks:
  - Extracting numbers from strings
  - Removing unwanted characters from text
  - Finding and replacing characters
  - Extracting specific parts of strings
  - Converting free form text to more uniform formats
  - Splitting strings into multiple values

* A function, parse_number() converts a number with commans (450,200) to a number without commas (450200) so the variable won't have a class of character and be a class of numeric.
* To show strings we can either use double quoutes ("10") or single qoutes ('10'). So, 10 inches can be represnted by this: ```s = '10"'``` and 5 feet: ```s = "5'"```. The slash (\) can be used to escape strings which can help us represent 5ft 10 inches: ```s = '5\'10"'```.
* In general, string proccesing requires strings and a pattern. Also, they consist of 4 parts, detecting, locating, extracting, and replacing elements of a string. For example, we need to remove commas from 4,500,323,102 so we need to detect the commas, locate them, extract them, and replace them with nothing ("").
* The stringr package from the tidyverse includes a variety of string processing functions that begin with str_ and take the string as the first argument, which makes them compatible with the pipe. [Stringr cheatsheet](https://evoldyn.gitlab.io/evomics-2018/ref-sheets/R_strings.pdf)
* Regular expressions (regex) is a way to describe specific characters of a text that can be used to determine if a given string matches the pattern, any string is a regex. A set of rules govern regex to make the proccess efficent and smooth as possible. Regex contains special characters to help search for certain types of strings (like digits), these special characters are shown in the cheatsheet. [Regex cheatsheet](https://rstudio.com/wp-content/uploads/2016/09/RegExCheatsheet.pdf).
* We can create strings to test our regex by making some to we know should match and some that shouldn't. Then, see if we get the expected results, this helps us check for 2 types of errors, failing to match and incorrectly matching. 
* Character classes are denoted by square brackets ([]). For example, to find strings/characters 3 or 4 we use [34] which can be written as: ```str_view(string, "[34")```. Or to find the numbers, 1 to 4 we can type: ```str_view(string, "[1-4]")```. Note: In regex everything is a character so 20 is not 20 its 2 or 0. 
* Anchors can be used to specify the start (^) and end ($) of a pattern. We can use anchors to specify exactly 1 digit, ^//d$ (//d is digits 0 - 9). This results in: ```str_view(string, "^//d$")```.
* Quantifiers can specify mulitple digits with curly brackets ({}) and possible number of times the pattern repeats. The pattern for 1 or 2 digit is //d{1,2}. This results in:```str_view(string, "^//d{1,2}$")```. Note: the asterik (*) means zero or more instances, which can be useful for quantifiers, ? = none or once, and + = one or more.
* Using all we learned we can create a regex to find a pattern of feet and inches in the format of: 4_to_7' 0_12" (an example being 5' 11"). The code for this would be: ```str_view(string, "^[4-7]'\\d{1,2}\"$")```.
* The things we disccused, regex, anchors, and quantifiers can be used for search and replace, via: ```str_replace(string, string_needed-to_be_replaced, string_replacing_it)```.
* Groups permit the extraction of values, being defined by parantheses (). Groups permit tools to identify specific parts of the pattern so it can be extracted. We want the first digit between 4 and 7 ([4-7]) with the second being none or more digits (\\d*) which results in "^[4-7], \\d*$". But, each is a group so we can encapsulate the parts we want to extract, resulting in: ```str_view(string, "^([4-7]), (\\d*)$")```. A powerful feature with groups is that you can refer to the extracted value in regex when searching and replacing, with \\\i finding the value from the ith group. \\\1 would be the value in the 1st group and \\\2 is the value in the 2nd group.
* Remember, sometimes it might not be worth writing code for some rare cases.
* The extract() function behaves similarly to the separate() function but allows extraction of groups from regular expressions:
```r
# first example - normally formatted heights
s <- c("5'10", "6'1")
tab <- data.frame(x = s)

# the separate and extract functions behave similarly
tab %>% separate(x, c("feet", "inches"), sep = "'")
tab %>% extract(x, c("feet", "inches"), regex = "(\\d)'(\\d{1,2})")

# second example - some heights with unusual formats
s <- c("5'10", "6'1\"","5'8inches")
tab <- data.frame(x = s)

# separate fails because it leaves in extra characters, but extract keeps only the digits because of regex groups
tab %>% separate(x, c("feet","inches"), sep = "'", fill = "right")
tab %>% extract(x, c("feet", "inches"), regex = "(\\d)'(\\d{1,2})")
```
Practice your regex [here](https://regexone.com/).

<strong>Dates, Times, and Text Mining:</strong>

Dates And Times:

* Dates can be represented as strings but once a reference day (epoch) is picked they can be converted to numbers. Most computer languages use January 1st, 1970 as the epoch. So every date can be a number, relative to the epoch (November 2nd, 2017 = 17,204). 
* But, the above can be confusing as if you were told 17,204 instead of Nov. 2nd, 2017 you would be quite confused. Similar things happen with time but even more confusing because of time zones. Because of this, a datatype just for dates and times is defined by R.
* Tidyverse has functionalities to deal with dates and time via the lubridate package ([cheatsheet](https://evoldyn.gitlab.io/evomics-2018/ref-sheets/R_lubridate.pdf)). Extract the year, month and day from a date object with the year(), month() and day() functions. The parsers convert strings into dates with the standard YYYY-MM-DD format (ISO 8601 format). Use the parser with the name corresponding to the string format of year, month and day (ymd(), ydm(), myd(), mdy(), dmy(), dym()).
* The Sys.time() gets the time in R with lubridate having a more advanced function, now(), allowing you to permit the time zone as a character ("PST"). You can extract values from time objects with the hour(), minute() and second() functions. OlsonNames() allows you to view all the time zones in lubridate and there are functions to parse strings into times ("12:34:56" -> "12H 34M 56S"). Parsers can also create combined date-time objects (for example, mdy_hms()).

Text Mining:

* Many applications data start out as text, like spam filtering, cyber-crime prevention, counter-terrorism, and sentiment analysis.
* An example of text minig is analyzing [Trump's tweets](http://varianceexplained.org/r/trump-tweets/). Here is a good [book](https://www.tidytextmining.com/) on text mining.
* The tidytext package helps convert from text to a tidy table. Having the data in this format helps the facilitation of data visualization and and applying statistical techniques.
* The main function needed to achieve this is unnest_tokens(). A token refers to the units that we are considering to be a data point. The most common tokens will be words, but they can also be single characters, ngrams, sentences, lines or a pattern defined by a regex. The functions will take a vector of strings and extract the tokens so that each one gets a row in the new table. Here is a simple example:
```r
example <- data_frame(line = c(1, 2, 3, 4),
                      text = c("Roses are red,", "Violets are blue,", "Sugar is sweet,", "And so are you."))
example
example %>% unnest_tokens(word, text)
```
* Collecting the most common words can result in just common words in the English language that aren't informative. The tidytext package has a database of these commonly used words, referred to as stop words (stop_words and stored as a dataframe), in text mining. These stop words can be filtered out to get a more accurate picture of the most common words. This can be accomplished with: ```vector_without_stop_words = vector_with_stop_words %>% filter(!word %in% stop_words$word)```.
* The count() function can be used to find the frequency of a word. Like this: ```count(char_vector, word) #Creates a table of words and how common they are.```
* In sentiment analysis we assign a word to one or more "sentiment". Although this approach will miss context dependent sentiments, such as sarcasm, when performed on large numbers of words, summaries can provide insights.
* The first step in sentiment analysis is to assign a sentiment to each word. The tidytext package includes several maps or lexicons in the object sentiments. There are several lexicons in the tidytext package that give different sentiments. For example, the bing lexicon divides words into positive and negative (get_sentiments("bing")). The AFINN lexicon assigns a score between -5 and 5, with -5 the most negative and 5 the most positive (get_sentiments("affin")). 
* After doing all this we find out that disgust, anger, negative sadness and fear sentiments are associated with the Android in a way that is hard to explain by chance alone. Words not associated to a sentiment were strongly associated with the iPhone source, which is in agreement with the original claim about hyperbolic tweets.
* An example of text mining:
```r
# We're going to be using the gutenbergr package which contains text files of many different books.
# Load everything up:
library(tidyverse)
library(gutenbergr)
library(tidytext)
options(digits = 3)
# The gutenberg_metadata contains all the books and documents.
# We'll use string_detect() to find the ID novel of Pride and Prejudice:
gutenberg_metadata %>%
    filter(str_detect(title, "Pride and Prejudice"))
# 6 different IDs were returned. 
# gutenberg_works() helps us filter out duplicate books and keep on english language works:
gutenberg_works(title == "Pride and Prejudice")$gutenberg_id
# This results in an ID number of 1342
book <- gutenberg_download(1342)
words <- book %>%
  unnest_tokens(word, text)
nrow(words)
# The above, downloads the book with ID 1342 and gets all the words in the book using unnest_tokens()
# Also, the nrow(words) prints out the number of words which is 122,204
words <- words %>% anti_join(stop_words)
nrow(words)
# The above, removes the stop words and stores it to the original vector (words). The number of words after removing stop words are 37246
words <- words %>%
  filter(!str_detect(word, "\\d"))
nrow(words)
# The above, removes any digits and stores it in words. The number of words after removing any digits are 37180
freq = words %>% count(word)
# The above, creates a data.frame of the frequency of the words in the book.
sum(freq$n > 100) 
# The above, finds the number of words that appear more than 100 times in the book using sum since freq$n > 100 returns a logical. 23 words appear more than 100 times in the book.
freq %>% filter(n == max(n))
# The above prints out the word that appears the most in the book and the number of times it appears. The word that appears the most in the book is elizabeth and it appears 597 times.
# Loads textdata which will help us use sentiment affin:
libaray(textdata)
afinn <- get_sentiments("afinn")
# The above, sets up a dataframe named afinn as the sentiments for afinn.
sent = inner_join(afinn, words)
nrow(sent)
# The above, sets up a data frame named sent (short for sentiment) as words that appear in both dataframe objects, afinn and words. nrow() finds out that 6,065 words have sentiments in the afinn lexicon.
mean(sent$value > 0)
# The above, finds out the proportion of words in the sent dataframe that have a positive value (basically positive sentiment). The proprotion is 0.563
sum(sent$value == 4)
# The above, finds the number of words that have a value of 4 in the sent data frame. Which are 51 words.
```
