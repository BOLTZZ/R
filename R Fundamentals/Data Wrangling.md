# Data Wrangling:
Data Import:
* One of the most common ways to store and share data is through spreadsheets (file version of a data frame).
* Spreadsheets have rows seperated by returns and columns separated by a delimiter. The most common delimiters are comma, semicolon, white space and tab.
* Many spreadsheets are raw text files and can be read with any basic text editor. However, some formats are proprietary and cannot be read with a text editor, such as Microsoft Excel files (.xls). Most import functions assume that the first row of a spreadsheet file is a header with column names. To know if the file has a header, it helps to look at the file with a text editor before trying to import it.
* The working directory is where R looks for files and saves files by default. On RStudio navigate to Session then Set Working Directory to set a working directory (it's suggested to create a new directory for each new project and keep the raw data in that directory).
