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

A file system can look like this:

![1](https://courses.edx.org/assets/courseware/v1/8a874934d6c335342808150e3be7a2d0/asset-v1:HarvardX+PH125.5x+1T2020+type@asset+block/data_science_1_rev.png)
* A filesystem is all files, folders, and programs on the computer.
