# Unix Notes:
An absolute/full path is the location of a file from the root directory. It's not dependent on the directory your working in right now and must begin with a / or ~. An example (full path to "project-1" file):

/home/projects/project-1

On the other hand, a relative path is the location of a file relative to your working directory. An example (working directory is "projects" folder and file is "projects-1"):

project-1

Some shortcuts:
1) One period “.” is your current working directory
2) Two periods “..” is the parent directory (up one from your present working directory) 
3) A tilde   “~” is your home directory.

A few examples:
1. Your current working directory is ~/projects and you want to move to the figs directory in the project-1 folder
  * Solution 1: cd ~/projects/project-1/figs (absolute)
  * Solution 2:  cd project-1/figs (relative)
2. Your current working directory is ~/projects and you want to move to the reports folder in the docs directory
  * Solution 1: cd ~/dos/reports (absolute)
  * Solution 2: cd ../docs/reports (relative)
3. Your current working directory is ~/projects/project-1/figs and you want to move to the project-2 folder in the projects directory.
  * Solution 1: cd ~/projects/project-2 (absolute)
  * Solution 2: cd ../../project-2 (relative)

File System:
* In Unix, we don't have the visual cues of directories and the file system. Though, the following notes will still be useful.
A file system can look like this:

![1](https://rafalab.github.io/dsbook/productivity/img/unix/filesystem.png)
* A filesystem is all files, folders, and programs on the computer.
* Folders are directories and folders in other folders are subdirectories (in the image, directory projects has 2 subdirectories, project-1 & project-2, with projects being a subdirectory of home). The home directory is your home directory (the name usually the user name). And, the root directory is where the whole filesystem originates from and is kept (the home directory is usally 2 or more levels from the root). The root directory is represented by a / in the Unix terminal.
* The working directory is the current directory we're in. pwd (print working directory) prints the current, working directory.
* Files (code, data, output) should be self-contained and structured.
* It is good practice to write a README.txt file to introduce the file structure to facilitate collaboration and for your future reference.

Some Advanced Unix:
* Arguments typically are defined using a dash (-) or two dashes (--) followed by a letter of a word.
* rm -r (r stands for recursive and files and directories are removed recursivley). rm -r \<directory_name> removes all files, subdirectories, files in subdirectories, subdirectories in subdirectories, etc, PERMANENTLEY.
* But, some directories can have protected files and have to use the force (-f) argument. Or can combine arguments to get: rm -rf \<directory-name> and remove directories regardless of protected files. <br>

Help Commands:
* Getting Help: Use man + command name to get help (e.g. man ls). Note that it is not available for Git Bash. For Git Bash, you can use command -- help (e.g. ls --help).
* Pipes: Pipes the results of a command to the command after the pipe. Similar to the pipe %>% in R. For example, man ls | less (and its equivalent in Git Bash: ls --help | less). Also useful when listing files with many files (e.g ls -lart | less). 

Wild Cards:
* The asterik (*) means any number of any combination of characters. Specifically, to list all html files: ls *.html and to remove all html files in a directory: rm *.html.
* Combined wild cards: rm file-001.* to remove all files of the name file-001 regardless of suffix.
* <strong>Warning: Combining rm with the * wild card can be dangerous. There are combinations of these commands that will erase your entire file system without asking you for confirmation. Make sure you understand how it works before using this wild card with the rm command.</strong>
* The question mark (?) means any single character. For example, to erase all files in the  form file-001.html with the numbers going from 1 to 999: rm file-???.html.

Enviroment Variables/Shells:
* Enviroment variables are settings in Unix that can affect command line settings, with the home directory being one of them.
* Variables are distinguished by adding a dollars sign ($) in front of the variable name (like $HOME for home directory). See all the variables by typing in env.
* You can change the variables but how you change them varies across different shells. Much of the commands covered here are on the Unix shell. Though, the differences across shells are very tiny. You can find what shell your using by typing in "echo $SHELL", the most common shell is bash.
* To change enviromental variables in the bash shell by typing in "export variable_name value_of_variable" and to change path type in "export PATH = /usr/bin/" (DON'T RUN THE PATH CHANGING COMMAND).
* All programs are files in Unix and the ones that run are called executables (ls, mv, git). To find where these files reside use the command which ('which git' would return user/bin/git).

Useful Unix Commands:
| Command | Description | Examples
| ---- | --- | --- 
| ls | List directory content | 
| mkdir dir | Make a directory | mkdir projects –make the directory projects <br> mkdir docs –make the directory docs <br> mkdir junk –make the directory junk
| rmdir dir | Remove a directory (directory must be empty; otherwise use “rm”) | rmdir junk –remove the directory junk
| cd dir | Change directory | cd /projects – move to the projects directory (an absolute path) <br> cd projects – move to the projects directory, assuming we are already in the home directory (a relative path)
| cd .. | Go up one directory to the parent directory| cd ../.. – move up two parent directories from our current directory
| cd ~ | Go to the home directory
| cd - | Go to whatever directory you just left
| pwd | Print the present working directory
| Tab key | Autocomplete | cd d + tab – autocompletes to docs if it is the only directory that begins with d; or list the different options.
| mv file1 file2 | Move or rename files <br> Warning –this is permanent, and you will not get a warning message if you are overwriting files. | mv ~/docs/resumes/cv.tex ~/docs/reports/ –move the cv.tex file from the resume folder to the reports folder <br> mv cv.tex resume.tex – rename cv.tex to resume.tex <br> mv ~/docs/resumes ~ /docs/reports/ - move the resume folder into the reports folder
| cp file1 file2 | Copy file1 to file2 | cp ~ ~/docs/reports/ – make a copy of the cv.tex file from the resume folder in the reports folder
| rm file | Delete file <br> Warning – this is permanent! You cannot retrieve files from the recycling bin! | rm ~/docs/resumes/cv.tex – delete the file cv.tex
| less file | View file | less ~/docs/resumes/cv.tex –open cv.tex in the less text viewer
| rm -r dir | Remove recursively all folders in directory dir and the directory itself.
| ls -a | List all directory content, including hidden files
| ls -l | List all directory content in long form (including permissions, size and date)
| ls -t | List all directory content in chronological order | ls -lart – show more information for all files in reverse chronological order for your current directory
| man command | Show the manual for the command. Note – this does not work for GitBash | man ls – show the manual instructions for the command ls.
| help | Show the manual for the command in GitBash | ls --help – show help instructions for the command ls
| command1 &#124; command2 | Pipe the results of command 1 to command 2 | man ls | less – show the help instructions for the command ls in the less viewer
| * (wildcard) | | ls *.html –list all the files ending in html in your current directory <br> rm *.html – remove all files ending in html in your current directory
|? (any character) | | rm file.???.html – remove all files whose names follow the pattern; eg file-001.html, file-002.html etc. <br> rm file.???.* – remove all files whose names follow the pattern regardless of their extension; eg file-001.html, file-002.csv, file-any.R, etc.
| $var | >$ identifies a variable | echo $HOME – print your home directory <br> echo $SHELL – print your shell name
| export val=value | Change the value of the variable val (Bash shell specific) | 
| open file (mac) start file (windows) | Opens a file or program | open Report.Rmd – opens Report.Rmd in RStudio on A MAC<br> start Report.Rmd - opens Report.Rmd in RStudio on A WINDOWS
| nano |  Sets up a bare-bones text editor | 
| ln | Create a symbolic link, use is not reccomended |
| tar | Archive files and subdirectories of a directory into one file |
| ssh  | Connect to another computer, pretty important |
| grep | search for patterns in a file |
| awk/sed | These are two very powerful commands that permit you to find specific strings in files and change them|
