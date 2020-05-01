*Note check [R Projects](https://github.com/BOLTZZ/R/tree/master/R%20Projects) for some good graphs.*
# Creating Graphs With Multiple Variables:
Dataset = Titanic.
Variables (Columns) = Age, Sex, Survived, Fare
Boxplots:
```r
# Boxplot of comparision of Survived and Fare.
titanic %>%
    filter(Fare > 0) %>%
    ggplot(aes(Survived, Fare)) +
    geom_boxplot() +
    scale_y_continuous(trans = "log2") +
    geom_jitter(alpha = 0.2)
```
Density plots:
```r
# The below is useful for comparing distribution:
titanic %>%
    ggplot(aes(Age, fill = Sex)) +
    geom_density(alpha = 0.2) +
    facet_grid(Sex ~ .)
# The below is useful for comparing count/number:
titanic %>%
   ggplot(aes(Age, y = ..count.., fill = Sex)) +
   geom_density(alpha = 0.2, position = "stack")
# The below shows accurate proportions:
titanic %>%
    ggplot(aes(Age, fill = Sex)) +
    geom_density(alpha = 0.2)
# Multiple graphs:
titanic %>%
    ggplot(aes(Age, y = ..count.., fill = Survived)) +
    geom_density(position = "stack") +
    facet_grid(Sex ~ Pclass)
```
Bar plots:
```r
# 2 variable plot (Survived and Sex):
#plot 1 - survival filled by sex
titanic %>%
    ggplot(aes(Survived, fill = Sex)) +
    geom_bar()
# plot 2 - survival filled by sex with position_dodge
titanic %>%
    ggplot(aes(Survived, fill = Sex)) +
    geom_bar(position = position_dodge())
#plot 3 - sex filled by survival
titanic %>%
    ggplot(aes(Sex, fill = Survived)) +
    geom_bar()
```
QQ plot:
```r
# QQ plot with theoretical line:
params <- titanic %>%
    filter(!is.na(Age)) %>%
    summarize(mean = mean(Age), sd = sd(Age))
titanic %>%
    filter(!is.na(Age)) %>%
    ggplot(aes(sample = Age)) +
    geom_qq(dparams = params) +
    geom_abline()
```
