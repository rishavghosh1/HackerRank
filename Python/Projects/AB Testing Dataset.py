#dataset: https://www.kaggle.com/datasets/zhangluyuan/ab-testing?select=ab_data.csv


import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import numpy as np



edadf= pd.read_csv("ab_data.csv")
edadf.head()
edadf.info()
edadf.nunique()
sns.countplot(data= edadf, x= 'group')

# We want to confirm that there are infact just 2 groups- control and treatment. We also notice that they have equal number of records for each group.

# We also want to see if there are users who are part of both groups. They need to be deleted as they would involve skewed results

newdf= edadf.drop_duplicates(subset ="user_id", keep = False)
sns.countplot(data= newdf, x= 'group', hue= 'landing_page')

conversion_rates = newdf.groupby('group')['converted'].agg(['count','mean','std'])
conversion_rates.head()


# We see that the mean conversion rate for the treatment group is actually less than the control group, however, the difference is very less. We have 12% conversion rate for control vs 11.9% conversion rate for Treatment. We can see that there is very little difference, however we need to check if this difference is statistically significant
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.weightstats import ztest as ztest

cdf = newdf[newdf['group'] == 'control']['converted']
tdf = newdf[newdf['group'] == 'treatment']['converted']

# The null hypothesis is that the difference between the 2 means is not statistically significant. p-value has to be greater than 0.05 to prove this.
# We chose 2 sample z test here as z-test is the statistical test, used to analyze whether two population means are different or not when the variances are known and the sample size is large.

ztest1=ztest(cdf,tdf,value=0)
ztest1


# The first number gives us the test statistic i.e. 1.194 and the second number gives us the p-value, which is .2. Therefore, we accept the null hypothesis and infer that the new changes didnot have any significant difference in conversion rate

# A z-statistic, or z-score, is a number representing the value’s relationship to the mean of a group of values, it is measured with population parameters such as population standard deviation and used to validate a hypothesis.
