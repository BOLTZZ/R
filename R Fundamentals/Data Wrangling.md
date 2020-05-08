# Data Wrangling:
Data Import:
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
* R-base import functions (read.csv(), read.table(), read.delim()) generate data frames rather than tibbles and character variables are converted to factors. This can be avoided by setting the argument stringsAsFactors=FALSE.
