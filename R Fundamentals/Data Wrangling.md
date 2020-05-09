# Data Wrangling:
<strong>Data Import:</strong>
* One of the most common ways to store and share data is through spreadsheets (file version of a data frame).
* Spreadsheets have rows seperated by returns and columns separated by a delimiter. The most common delimiters are comma, semicolon, white space and tab.
* Many spreadsheets are raw text files and can be read with any basic text editor. However, some formats are proprietary and cannot be read with a text editor, such as Microsoft Excel files (.xls). Most import functions assume that the first row of a spreadsheet file is a header with column names. To know if the file has a header, it helps to look at the file with a text editor before trying to import it.
* The working directory is where R looks for files and saves files by default. On RStudio navigate to Session then Set Working Directory to set a working directory (it's suggested to create a new directory for each new project and keep the raw data in that directory).
* readr is a library of the tidyverse package that contains functions for reading data stored in text file spreadsheets into R:

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
* The join functions in the dplyr package combine two tables such that matching rows are together.

<img src = "https://rafalab.github.io/dsbook/wrangling/img/joins.png" width = 300 height = 350>

(Image courtesy of RStudio. CC-BY-4.0 license. Cropped from original.)
