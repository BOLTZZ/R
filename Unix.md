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
