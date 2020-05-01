# Murders Graph Using ggplot():
Output:

![1](https://github.com/BOLTZZ/R/blob/master/Images%26GIFs/Rplot.png)

Code:
```r
library(tidyverse)
# Loads the tidyverse package. This will help us access ggplot().
library(dslabs)
# Loads the dlsabs package. This will help us access the murders dataset.
library(ggtheme)
# Loads the ggtheme package. This will help us set up the theme of the graph.
library(ggrepel)
# Loads the ggrepel package. This will help us make sure the abberviation labels don't fall on top of each other using a geometry.
data(murders)
# Gets murders data set
r <- murders %>%summarize(rate = sum(total) / sum(population) * 10^6) %>% pull(rate)
# We find the value of r which is the ratio of all murders to population which can be accomplished via the summarize function. The r is the slope of the line of average murder rate.
murders %>%
  ggplot(aes(population/10^6, total, label = abb)) + 
# Also, a global object of asthetic mappings (aes) is defined so we don't have to keep on adding aes to each layer. But, if needed each layer can redefine aes.
# And, the x axis is population/10^6 and y axis is total.  
  geom_abline(intercept = log10(r), lty = 2, color = "darkgrey") +
  geom_point(aes(col = region), size = 3)
# This recolors the points based on region and creates a legend. Also, using geom_abline we create a line for average murder rate.
  geom_text_repel() +
# This time, we add another layer to the graph. This layer adds the abbreviation to each point of the scatterplot/data. Also, we use repel 
# because its a function that uses a special geometry to make sure the state abbreviations don't cover each other. NOTE: geom_text adds text without a rectangle unlike geom_label. 
# Another NOTE: We don't use murders$population or murders$total since aes knows we are piping in data from murders. 
# But, if we take the population out of aes we will get an error since its not a globally defined variable.
  scale_x_log10() +
  scale_y_log10() +
# The above lines scale down the units and the x and y coordinates by a factor of log10.
  xlab("Population in millions (log scale)") +
  ylab("Total number of murders (log scale)") +
  ggtitle("US Gun Murders in 2010:") +
# The above adds the axes titles and the graph title.
  scale_color_discrete(name = "Regions:") +
# Changes the title of the legend.
  theme_economist()
# Changes the theme of the graph to a theme like the economist.
```
