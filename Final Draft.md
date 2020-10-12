
# Time Series Final Project SYS 7030 -- Final Draft

**Author:** Qimin Luo
**Date:** 10/11/2020

##  *Abstract*
Netflix is one of the most successful video websites in the United States. Futher, it is also the most profitable video website in the world. It is interesting to explore its future development. In this experiment, we apply Prophet procedure to build a time series model and make preditions about its future market cap. 

## Introduction
Netflix is already one of the biggest media companyies in the world. Depsite this, there are some fast-growing media companies like Hulu, Youtube is running after it. We are interested to explore its future development. We plan to apply Prophet procedure to build a time series model to predict its future market cap. Its future development is impossible to be determined only by market cap. However, it is expected to find out some clues about future car development trend. 


## Data Generation
It is a pretty complicated and time-consuming process to retrive and clean data from Internet manully. In this case, I use a convenient python financial data package -- Quandl. It contains almost unlimited financial data. Further, it can provide us with clear formated data.

![jupyter](dataformat.png)

Based on quandl, we can retrive data and build a plot below to visualize it.

![jupyter](visual.jpg)

### Data Generating Process

The market cap of a company can be affected by lots of variables. It is reason take variables like daily stock prices and stock vlomue into account. However, some social and political event can also have crucial effecs on market cap. We are not able to add these things into model so we have to skip these unpredictable factors. In this case, it is clear that variables like open price, daily highest price, daily lowest price and close price have directive influence on its market cap. Further, stock volume is also important because it is not fair to determine market cap by its stock price. For example, some companies have very high stock prices but its stock volume is small. In this case, its market cap is limited by volume. Addtionally, we conisder some adjusted factors like Adj. High since adjusted variables sometimes can demonstrate information clearly. And companies' actions like releasing a new product can change their market cap.


### Formal Model

I decide to apply Prophet Model which is explored by Facebook. Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data.

Prophet Model can be roughly represented by these formulas below.

$ y(t) = g(t) + h(t) + s(t) + \epsilon{(t)} $

$ g(t) = (k + \alpha (t)\delta)\cdot t+(m+\alpha(t)^{T}\gamma)$ （1）

$ s(t) = \sum_{n=1}^{N}(a_{n}cos(\frac{2\pi nt}{p}) + b_{n}sin(\frac{2\pi nt}{p})) $ （2）

$ h(t) = Z(t) \textbf{k} $ （3）

$ Z(t) = [1(t\in D_{1}),...,1(t\in D_{L})], \textbf{k} = (k_{1},...,k_{L})^{T}$

(1), (2), (3) is repectively represent trend, seasonality and holidays and events.

### Model Discussion

Prophet is optimized for the business forecast tasks , which typically have any of the following characteristics:

- hourly, daily, or weekly observations with at least a few months (preferably a year) of history
- strong multiple “human-scale” seasonalities: day of week and time of year
- important holidays that occur at irregular intervals that are known in advance (e.g. the Super Bowl)
- a reasonable number of missing observations or large outliers
- historical trend changes, for instance due to product launches or logging changes
- trends that are non-linear growth curves, where a trend hits a natural limit or saturates

Market Cap meets most of above properties so I think Prophet might be suitable in this experiments. At its core, the Prophet procedure is an additive regression model with four main components:

- A piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting changepoints from the data.
- A seasonal component modeled using Fourier series.
- A user-provided list of important holidays.

These three components can fit data more flexibly since market caps are easily affected by some seaonal events and holidaies. For instance, more people watch videso at holidaies and weekends. Moreover, Prophet is good at dealing with missing data. Therefore, I state Prophet might be better than traditional models.

## Plan for Data Analysis

Thorugh this experiment, I expect to build a model to predict future trend of Netflix's market cap. It is not feasible to create a tool to make precise preditions about its market cap because there are so many factors we can't quantify and control.
Despite this, we are able to have a model to make predictions. Based on these predictions, future trend of its market cap is likely to estimated. In this experiment, we could apply fbprophet python package from Facebook to build models and make predictions. Additionally, we might need packages like pandas and matplotlib to help analyze and visualize data.

## Reference


- [1] Ignacio Medina, David Montaner, Joaquín Tárraga, Joaquín Dopazo, Prophet, a web-based tool for class prediction using microarray data, Bioinformatics, Volume 23, Issue 3, 1 February 2007, Pages 390–391
