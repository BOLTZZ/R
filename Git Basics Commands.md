# Notes:

<strong>Four stages:</strong><br>
Remote Stage:<br>
  * Upstream repository - All the files you see once you access a repository on Github. The git fetch command fetches the upstream repositor to your local repository. And, git merge adds the files from local repo. to the working directory and stagign area. But, git pull does git fetch and git command.<br>

Local Stages:<br>
  * Working directory - Whatever file/directory your currently editing on an ide with Git as version control (same as Unix working directory). <br>
  * Staging Area - Edits in the staging area are not kept by the version control, git add adds the files from our working directory to the staging area. The staging command can be skipped by directly commiting from working directory to local repository.<br>
  * Local repository - The local repository stores all the files, locally, git commit commits the files from the staging area to the local repository. The push command (git push) pushes all the work to the upstream repository (Github repo).<br>

Some basic commands:
1) git status - shows the status of any additions/modifications/deletions
2) git add* - adds the files to local
3) git commit -m"some comment" - commits to local respository
4) git push origin master - finally they are in your github account
5) git pull - your files are updated to changes other people made
6) git log - keeps track of all the changes we have made to the local repository

Creating repos:<br>
* One way to create a repo is by using the git clone command to clone an existing repo from Github to your computer. <br>
* Another way is to finish your project on your computer. Then, create an upstream repo and call git init (to initalize) on your working directory (project) so the files will be tracked by Git. Now, just commit (or add and then commit) the files from the working directory to your local directory. Time to connect the upstream repo and local directory by "git remote add origin <upstream-rep-url>". Note: The first time you push to a new repository, you may also need to use these git push options: git push --set-upstream origin master. If you need to run these arguments but forget to do so, you will get an error with a reminder. <br>

[Resource by Github](https://try.github.io/)
