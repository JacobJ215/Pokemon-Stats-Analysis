## Pokemon Stats Analysis and ML Project
![](images/Pokemon%20Analysis%20Dashboard.png)
[Link to Tableau Dashboard](https://public.tableau.com/views/pokemon_16807219032990/PokemonAnalysisDashboard?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link)

This is a data analysis and machine learning project that focuses on analyzing the stats of Pokemon from the popular Pokemon game series. The project utilizes Python and various data analysis libraries, including Pandas, Numpy, Seaborn, and Matplotlib, to explore and visualize the data, as well as perform statistical analysis on the Pokemon stats.


### Exploratory Data Analysis
The project performs an exploratory data analysis (EDA) on the Pokemon data to gain insights and understand the structure of the data. This includes checking for null values and data types using the info() function, calculating the percentage of null values using the isnull().mean() function, and displaying summary statistics of the data using the describe() function.

### Statistical Analysis
The project performs statistical analysis on the Pokemon data to calculate various statistics, including the mean, median, and standard deviation of different attributes such as Attack, Defense, HP, and Speed.

#### Statistical Analysis Results

I conducted a t-test to compare the Attack stat between different generations of Pokemon. The results are as follows:

Generation 1 vs. Generation 2:

* t-statistic: 1.1763035045816723
* p-value: 0.24050968418101457


Based on a significance level of 0.05, we do not have enough evidence to reject the null hypothesis, as the p-value (0.24050968418101457) is greater than the significance level. Therefore, there is no statistically significant difference in the Attack stat between Generation 1 and Generation 2 of Pokemon.

Generation 1 vs. Generation 3:

* t-statistic: -1.333905980818188
* p-value: 0.18317154297248378

Similarly, for Generation 1 vs. Generation 3, we do not have enough evidence to reject the null hypothesis, as the p-value (0.18317154297248378) is greater than the significance level of 0.05. Therefore, there is no statistically significant difference in the Attack stat between Generation 1 and Generation 3 of Pokemon.

Conclusion:

These results suggest that there is no significant difference in the Attack stat between the different generations of Pokemon based on the conventional significance level of 0.05 commonly used in hypothesis testing. 

### Data Visualizations
The project uses data visualizations to explore the relationships between different variables and to visualize the distributions of Pokemon stats. This includes creating pair plots using the pairplot() function from Seaborn, creating boxplots and scatter plots using the boxplot() and regplot() functions from Seaborn, and creating heatmaps using the heatmap() function from Seaborn.

The project also groups the Pokemon data by 'Type 1' and 'Type 2' and calculates the statistical averages for different attributes. It then creates barplots to visualize the average Pokemon stats for each type.

![](images/feature_dist.png)


As we can see with this box plot our stats are fairly evenly distributed with most IQRs ranging from 50-100. We can also see the HP as the most amount of outliers but also has the smallest "maxiumum" and "IQR"

![](images/attack_v_defense.png)


With the chart we can see that there is a positive linear relationship between Attack and Defense

Here are the results from our correlation matrix

![](images/correlation_matrix.png)


Below are the average stats compared across the primary and secondary pokemon types

![](images/stat_attributes_by_type_1.png)

![](images/stat_attributes_by_type_2.png)

#### Some additional Findings
![](images/top_10_attack.png)


![](images/top_10_attack_non_legend.png)

### Predictive Analysis using Machine Learning


The project will include a section for predictive analysis using machine learning to predict the Legendary status of Pokemon. This section uses machine learning techniques such as classification algorithms to train a model on the Pokemon data and make predictions on whether a Pokemon is Legendary or not.