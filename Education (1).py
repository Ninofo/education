#!/usr/bin/env python
# coding: utf-8

# ## Education Project
# Can socioeconomic factors predict ACT scores?

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


# In[6]:


edgap= pd.read_excel(r"C:\Users\Owner\Documents\Graduate School\Fall 2026 Quarter\DATA 5100- Foundation of Data Science\Education Project 5100\data-education\EdGap_data.xlsx", dtype = {'NCESSCH School ID':object})


# In[8]:


school_information= pd.read_csv(r"C:\Users\Owner\Documents\Graduate School\Fall 2026 Quarter\DATA 5100- Foundation of Data Science\Education Project 5100\data-education\ccd_sch_029_1617_w_1a_11212017.csv", encoding = 'unicode_escape')


# In[9]:


edgap.head()


# In[10]:


school_information.head()


# In[11]:


pd.set_option('display.max_columns',None)
school_information.head()


# In[12]:


edgap.info()


# In[13]:


school_information.info()


# ## Inital assessment of Data Frames
# - School_info dataset is huge compared to Edgap dataset
# - Edgap data set doesnt include all the schools. 
# - Each dataset has missing values but is tidy or long format
# - School identifier is an object in Edgap but a float in school_info data set

# # Is Data Sufficient For Answering The Question?
# Working with regression analysis to see relationship of ACT score vs. various socioeconomic factors
# pairplot- shows scatter plots of all pairwise relationships between variables AND histograms of individiaul variables(to see distribution)

# In[14]:


sns.pairplot(
    edgap.drop(columns="NCESSCH School ID")
);
plt.show()


# In[15]:


fig = sns.pairplot(
    edgap.drop(columns="NCESSCH School ID"), 
    kind = "reg", 
    plot_kws={
        "line_kws":{"color":"blue"},
        "scatter_kws": {"alpha": 0.5, "color":"k", "s":7},
    },
)

for ax in fig.axes.flat:
    if ax.get_xlabel() == "CT Median Household Income":
        ax.ticklabel_format(style = 'sci', axis = 'x', scilimits=(0,0))
    ax.set_xlabel(ax.get_xlabel(), fontsize = 8, rotation = 30, ha = 'right')
    ax.set_ylabel(ax.get_ylabel(), fontsize = 8)

    plt.setp(ax.get_xticklabels(), rotation=30, ha ='right')

plt.show()


# In[16]:


fig = sns.pairplot(
    edgap.drop(columns="NCESSCH School ID"), 
    y_vars=['School ACT average (or equivalent if SAT score)'],
    kind = "reg", 
    plot_kws={
        "line_kws":{"color":"blue"},
        "scatter_kws": {"alpha": 0.5, "color":"k", "s":7},
    },
)

for ax in fig.axes.flat:
    if ax.get_xlabel() == "CT Median Household Income":
        ax.ticklabel_format(style = 'sci', axis = 'x', scilimits=(0,0))
    ax.set_xlabel(ax.get_xlabel(), fontsize = 8, rotation = 30, ha = 'right')
    ax.set_ylabel(ax.get_ylabel(), fontsize = 8)

    plt.setp(ax.get_xticklabels(), rotation=30, ha ='right')

plt.show()


# # Results
# - There appears to be relationship betwee socioeconomic variables and ACT score.
# - There are some out of range ACT and percent lunch values 
# - All in all data set is a good candidate to answer the question

# ## Data Prep

# In[18]:


school_information = school_information[
    ['SCHOOL_YEAR', 'NCESSCH','LSTATE', 'LZIP', 'SCH_TYPE_TEXT', 'LEVEL', 'CHARTER_TEXT']
    ]


# In[19]:


school_information = school_information.rename(
    columns={
        "SCHOOL_YEAR": 'year',
        "NCESSCH": 'id',
        "LSTATE": 'state',
        "LZIP": 'zip_code',
        "SCH_TYPE_TEXT": 'school_type',
        "LEVEL": 'school_level',
        "CHARTER_TEXT": 'charter'
    }
)


# In[20]:


edgap = edgap.rename(
    columns = {
        "NCESSCH School ID": "id",
        "CT Pct Adults with College Degree": "percent_college",
        "CT Unemployment Rate": "rate_unemployment",
        "CT Pct Childre In Married Couple Family": "percent_married",
        "CT Median Household Income": "median_income",
        "School ACT average (or equivalent if SAT score)": "average_act",
        "School Pct Free and Reduced Lunch": "percent_lunch",
    }
)



# In[21]:


edgap.head()


# In[22]:


school_information.head()


# ## Join Data Frames and Quality Control

# In[24]:


school_information['id'] = school_information['id'].astype('object')


# In[25]:


school_information.info()


# In[26]:


df= edgap.merge(
    school_information, 
    how= 'left',
    on = 'id'
)


# In[27]:


df.head()


# In[28]:


df.info()


# In[29]:


df.describe()


# In[30]:


df.select_dtypes(include = ['number']).agg(['min', 'max']).round(2)


# In[31]:


df.loc[df['percent_lunch'] <0, 'percent_lunch'] = np.nan
df.loc[df['average_act'] <1, 'average_act']=np.nan


# In[32]:


df['school_type'].value_counts()


# In[33]:


df['school_level'].value_counts()


# In[34]:


df['charter'].value_counts()


# In[35]:


df= df.loc[df['school_level'] == 'High']


# In[36]:


df.duplicated().sum()


# In[37]:


df.isna().sum().to_frame(name='Number of Missing Values')


# In[38]:


percent_missing = df.isna().mean().round(4)*100
percent_missing.to_frame(name='Percent Missing Values')


# In[39]:


df['state'].value_counts()


# In[40]:


df['state'].nunique()


# In[41]:


import plotly.offline as po
import plotly.graph_objs as pg


# In[42]:


layout = dict(
    geo={"scope": "usa"}, coloraxis_colorbar = dict (title = "Number of Schools")
)
data = dict(
    type = "choropleth",
    locations = df ["state"].value_counts().index,
    locationmode= "USA-states",
    z= df["state"].value_counts().values,
    coloraxis="coloraxis",
)
x = pg.Figure(data = [data], layout =layout)
po.iplot(x)


# In[43]:


df=df.dropna(subset=['average_act'])


# In[44]:


df.isna().sum().to_frame(name='Number of Missing Values')


# Impute Missing values in Socioeconomic Variables

# predictor_variables= [
#     'rate_unemployment',
#     'percent_college',
#     "percent_married",
#     'median_income',
#     'percent_lunch',
#     'state',
#     'charter'
# ]

# In[46]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[47]:


imputer = IterativeImputer()


# In[48]:


numerical_predictors = df[predictor_variables].select_dtypes(include ='number').columns.to_list()
print(numerical_predictors)


# In[49]:


imputer.fit(df.loc[:, numerical_predictors])


# In[50]:


df.loc[:,numerical_predictors]=imputer.transform(df.loc[:,numerical_predictors])


# In[51]:


df.isna().sum().to_frame(name='Number of Missing Values')


# In[52]:


df.to_csv(
    'education_clean.csv',
    encoding = 'utf-8-sig',
    index = False
)


# ## Exploratory Analysis

# In[92]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.anova import anova_lm


# In[93]:


df.head()


# In[95]:


predictor_variables = ['rate_unemployment', 'percent_college', 'percent_married', 'median_income', 'percent_lunch','state']
numerical_predictors = df[predictor_variables].select_dtypes(include = 'number').columns.to_list()
corr_matrix = df[numerical_predictors + ['average_act']].corr()
plt.figure(figsize = (10,5))
sns.heatmap(
    corr_matrix, vmax=1, vmin = -1, square = True, annot = True, cmap = 'viridis'
)
plt.tick_params(labelsize = 12)
plt.show()


# In[97]:


fig = sns.pairplot(
    data =df, 
    vars=numerical_predictors + ['average_act'],
    hue = 'charter',
    kind = 'reg',
    plot_kws = {"scatter_kws":{'alpha':0.5, "color": 'k','s':7},
               },
)

for ax in fig.axes.flat:
    if ax.get_xlabel() == 'CT Median Household Income':
        ax.ticklabel_format(style = 'sci', axis ='x', scilimits = (0,0))
    ax.set_xlabel(ax.get_xlabel(), fontsize =8, rotation = 30, ha = 'right')
    ax.set_ylabel(ax.get_ylabel(), fontsize = 8)

    plt.setp(ax.get_xticklabels(), rotation = 30, ha = 'right')
plt.show()


# In[99]:


plt.figure(figsize =(10,3))
fractions = list (numerical_predictors)
fractions.remove('median_income')

sns.boxplot(data= df[fractions], color = 'k')
plt.ylabel('Proportion', fontsize = 15)
plt.tick_params(labelsize =12)
plt.show()


# In[101]:


plt.figure(figsize = (3,3))

sns.boxplot(data = df, y = 'median_income', color = 'k')
plt.ylabel('Median income ($)')

plt.show()


# numerical predictors appear to be related to the average ACT scores and their relationships look linear at first glance.Several predictors appear to be correlated with each other as well. 

# ## Regression Modeling 

# In[53]:


plt.figure(figsize=(6,6))

sns.regplot(data = df,
            x= 'median_income',
            y= 'average_act',
            color = 'blue',
            ci = False,
            scatter_kws= {'color': 'black', 'edgecolors': 'white', 'linewidths': 1})

plt.xlabel('Median Income ($)', fontsize = 16)
plt.ylabel('Average ACT Score', fontsize=16)

plt.tick_params(labelsize=14)

plt.show()


# In[54]:


import statsmodels.formula.api as smf
model_median_income = smf.ols(formula= 'average_act ~ median_income', data=df).fit()


# In[55]:


print(model_median_income.summary())


# In[56]:


#compute R-squared
model_median_income.rsquared


# In[57]:


#compute the Root Mean Square Error (RMSE)
y_hat = model_median_income.predict()


# In[58]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(df['average_act'], y_hat)).round(3)


# In[59]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(df['average_act'], y_hat)


# In[60]:


plt.figure(figsize = (6,6))

plt.plot(y_hat, model_median_income.resid, 'ko',mec = 'w')
plt.axhline(0,color ='r', linestyle = 'dashed', lw = 2)

plt.xlabel('Predicted average ACT score', fontsize = 16)
plt.ylabel('Residual average ACT score', fontsize = 16)

plt.tick_params(labelsize = 14)

plt.show()


# In[61]:


plt.figure(figsize=(6,6))

sns.regplot(data = df,
            x= 'median_income',
            y= 'average_act',
            color = 'blue',
            ci = False,
            scatter_kws= {'color': 'black', 'edgecolors': 'white', 'linewidths': 1})

sns.regplot(data = df, 
            x = "median_income",
            y= "average_act",
            order = 2, 
            color = 'orange', 
            ci = False, 
            scatter = False)

plt.xlabel('Median Income ($)', fontsize = 16)
plt.ylabel('Average ACT Score', fontsize=16)

plt.tick_params(labelsize=14)

plt.show()


# In[62]:


#fit quadratic linear regression model
model_median_income2 = smf.ols(
    formula ='average_act ~ median_income + I (median_income**2)',
    data = df).fit()


# In[63]:


print(model_median_income2.summary())


# In[64]:


model_median_income2.pvalues


# In[65]:


from statsmodels.stats.anova import anova_lm
anova_lm(model_median_income, model_median_income2)


# In[66]:


#compute mean abs erro
mean_absolute_error(df['average_act'],model_median_income2.predict())


# In[67]:


plt.figure(figsize=(6,6))

sns.regplot(data = df,
            x= 'percent_college',
            y= 'average_act',
            color = 'green',
            ci = False,
            scatter_kws= {'color': 'black', 'edgecolors': 'white', 'linewidths': 1})

plt.xlabel('Percent College (%)', fontsize = 16)
plt.ylabel('Average ACT Score', fontsize=16)

plt.tick_params(labelsize=14)

plt.show()


# In[102]:


plt.figure(figsize=(6,6))

sns.regplot(data = df,
            x= 'rate_unemployment',
            y= 'average_act',
            color = 'green',
            ci = False,
            scatter_kws= {'color': 'black', 'edgecolors': 'white', 'linewidths': 1})

plt.xlabel('Rate Unemployment (%)', fontsize = 16)
plt.ylabel('Average ACT Score', fontsize=16)

plt.tick_params(labelsize=14)

plt.show()


# #Multiple Linear regression

# In[68]:


model = smf.ols(
    formula = 'average_act ~ rate_unemployment + percent_college + percent_married + median_income + percent_lunch',
    data= df).fit()


# In[69]:


print(model.summary())


# In[70]:


y_hat = model.predict()


# In[71]:


plt.figure(figsize = (5,5))

plt.plot(y_hat, model.resid, 'ko', mec= 'w')
plt.axhline(0, color = 'r', linestyle = 'dashed', lw = 2)

plt.xlabel('Predicted average ACT score', fontsize = 16)
plt.ylabel('Residual average ACT score', fontsize = 16)

plt.tick_params(labelsize = 14)

plt.show()


# In[72]:


mean_absolute_error(df['average_act'], model.predict())


# multiple linear regression model with ALL socioeconomic predictors CAN predict ACT scores with an average error of about 1 point using these predictors
# residual plot suggests there will be limited improvement from adding transformations of these predictors
# 2x predictors did NOT have significant coefficients
# adding predictors to model may improve performance

# In[73]:


#reduced model with siginificant predictors
model_reduced = smf.ols(
    formula = 'average_act ~ rate_unemployment + percent_college + percent_lunch', 
    data = df).fit()


# In[74]:


print(model_reduced.summary())


# In[75]:


y_hat = model_reduced.predict()


# In[76]:


plt.figure(figsize = (5,5))

plt.plot(y_hat, model_reduced.resid, 'ko', mec='w')
plt.axhline(0, color = 'r', linestyle = 'dashed', lw=2)

plt.xlabel('Predicted average ACT score', fontsize = 16)
plt.ylabel('Residual average ACT score', fontsize = 16)

plt.tick_params(labelsize = 14)

plt.show()


# In[77]:


mean_absolute_error(df['average_act'], model_reduced.predict())


# In[78]:


mae_full = mean_absolute_error(df['average_act'], model.predict())
mae_reduced = mean_absolute_error(df['average_act'], model_reduced.predict())

r2_full = model.rsquared
r2_reduced = model_reduced.rsquared

pd.DataFrame({'Mean Absolute Error': [mae_full, mae_reduced],
              'R-squared':[r2_full, r2_reduced]},
             index = ['full model', 'reduced model']).round(4)


# In[79]:


anova_lm(model_reduced, model)


# #Scaling reduced model 
# for mean =0  and st dev.= 1

# In[80]:


predictor_variables = ['rate_unemployment', 'percent_college', 'percent_lunch']


# In[81]:


scaled_columns = [var + '_normalized' for var in predictor_variables]
print(scaled_columns)


# In[82]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler().fit(df[predictor_variables])


# In[83]:


df[scaled_columns] = scaler.transform(df[predictor_variables])


# In[84]:


df[scaled_columns].agg(['mean', 'std']).round(3)


# In[85]:


model_normalized = smf.ols(
    formula = 'average_act ~ rate_unemployment_normalized + percent_college_normalized + percent_lunch_normalized',
    data = df).fit()


# In[86]:


print(model_normalized.summary())


# In[87]:


mae_normalized = mean_absolute_error(df['average_act'], model_normalized.predict())
mae_reduced = mean_absolute_error(df['average_act'], model_reduced.predict())

r2_normalized = model_normalized.rsquared
r2_reduced = model_reduced.rsquared

pd.DataFrame({'Mean Absolute Error':[mae_normalized, mae_reduced],
              'R-squared':[r2_normalized, r2_reduced]},
             index = ['noramlized model', 'reduced model']).round(4)


# results
# percent of students receiving reduced-price lunch at school is the strongest predictor for ACT score in this analysis
# 
