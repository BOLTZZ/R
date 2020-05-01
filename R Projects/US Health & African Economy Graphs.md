# US Health & African Economy Graphs:
Density plot of dollars per day in Africa in 1970 and 2010:

![1](https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/africa.PNG)

Three variable (years, states, smallpox cases) graph of U.S. smallpox cases (Time series plot):

Output:

![2](https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/california.png)

Code:
```r
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(dslabs)
data(us_contagious_diseases)

the_disease = "Smallpox"
dat <- us_contagious_diseases %>% 
   filter(!state%in%c("Hawaii","Alaska") & disease == the_disease & weeks_reporting >= 10) %>% 
   mutate(rate = count / population * 10000) %>% 
   mutate(state = reorder(state, rate))
dat %>% ggplot(aes(year, state, fill = rate)) + 
  geom_tile(color = "grey50") + 
  scale_x_continuous(expand=c(0,0)) + 
  scale_fill_gradientn(colors = brewer.pal(9, "Reds"), trans = "sqrt") + 
  theme_minimal() + 
  theme(panel.grid = element_blank()) + 
  ggtitle(the_disease) + 
  ylab("States:") + 
  xlab("Year:")
```
California Disease Rates per 10,000 (Time series plot):

Output:

![3](https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/california%20disease.png)

Code:
```r
library(dplyr)
library(ggplot2)
library(dslabs)
library(RColorBrewer)
data(us_contagious_diseases)

us_contagious_diseases %>% filter(state=="California" & weeks_reporting >= 10) %>% 
  group_by(year, disease) %>%
  summarize(rate = sum(count)/sum(population)*10000) %>%
  ggplot(aes(year, rate, color = disease)) + 
  geom_line() + ggtitle("California Disease Rate per 10,000:")
```
US Disease Rates per 10,000 (Time series plot):

Output:

![4](https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/US_disease.png)

Code:
```r
library(dplyr)
library(ggplot2)
library(dslabs)
library(RColorBrewer)
data(us_contagious_diseases)
dat = us_contagious_diseases %>% filter(!is.na(population)) %>% group_by(year, disease) %>% summarize(rate = sum(count)/sum(population) * 10000) %>% ggplot(aes(year, rate, color = disease)) + geom_line() + ggtitle("US Disease Rate per 10,000:")
dat #prints dat out.
```
