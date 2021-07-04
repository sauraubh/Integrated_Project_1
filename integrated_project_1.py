#!/usr/bin/env python
# coding: utf-8

# # Project: Integrated Project Module 1

# Goal of the Project: Forecasting sales for 2017 sales based on previous data.

# ## 1.Open the data file and study the general information

# In[1]:


import pandas as pd
import numpy as np 
import os
import pylab as pl
import matplotlib.pyplot as plot

try:
    df = pd.read_csv('C:/Users/Asus/Downloads/games.csv')
except:
    df = pd.read_csv('/datasets/games.csv')
print(df.info())
print(df.describe())
df


# Opened the given datafile and looked for general inforamtion of the table.
# 
# Problem with data:
# 1. Lots of missing values
# 2. Unnecessary data
# 3. datatypes are not reformed

# # 2.Prepare the data

# ## 2.1 Replace the column names (make them lowercase). 

# In[2]:


print(df.columns)
df.columns= df.columns.str.lower()
df.columns


# ## 2.2 Working on Missing Values. 

# In[3]:


from io import BytesIO
import requests
spreadsheet_id = "1zLDcpYndMunIBPl9NV_c_KGW7irAvuBSER41GkRp9uQ"
file_name = 'https://docs.google.com/spreadsheets/d/{}/export?format=csv'.format(spreadsheet_id)
r = requests.get(file_name)
df2 = pd.read_csv(BytesIO(r.content),index_col=None)
df2


# We are using another dataset from kaggle to fill some missing values in project. 

# In[4]:


df2 = df2.dropna()
df2 = df2.reset_index(drop=True)
df2.isnull().sum()


# We processed the data only which we needed from our second dataframe so, here we get rid of unwanted data.

# In[5]:


df["name"].fillna("No Name", inplace = True) 
df["genre"].fillna("No genre", inplace = True) 


# we have only two missing values in name and genre column which we gonna fill with No name and No genre for now we are gonna deal with it later

# In[6]:


df2['released'] = pd.to_datetime(df2['released'],format='%Y-%m-%d')
df2['released'] = pd.DatetimeIndex(df2['released']).year


# In[7]:


release_dict = pd.Series(df2["released"].values,index=df2["name"]).to_dict()
df['year_of_release'] = df['year_of_release'].fillna(df['name'].map(release_dict))
df['year_of_release'].isnull().sum()


# we tried to fill missing values for year of release from our second dataframe but we got still some missing values as ma be some names did not match similarly with our original dataset.

# In[8]:


df['year_of_release_noempty']= df.groupby('genre')['year_of_release'].transform(lambda y : y.fillna(y.mode()[0]))

df['year_of_release_noempty'].isnull().sum()


# We are gonna fill those missing years with mode as we are gonna use most common year in those perticular genre to fill missing values. we have only 269 missing values which are less than 1% of what we have so, I don't think it is going to affect our data and further callculation.
# 
# Mode is the most frequently occuring value in a dataset or distribution. A dataset can have more than one mode. which here help us to get most frequent year to fill up for that perticular genre.

# In[9]:


df['critic_score_noempty']= df.groupby('genre')['critic_score'].transform(lambda y : y.fillna(y.mean()))
df["critic_score_noempty"].fillna("0", inplace = True) 
df['critic_score_noempty'].isnull().sum()


# Critic Score missing: It may be because critics don't want to rate the game as it may be released for short time or it may be not that popular so we are filling it with mean of grouping it with genre so, it should not affect data. and we got two missing values which we are gonna fill with 0 as those values for No name and No genre

# In[10]:


df.loc[df['user_score'] == 'tbd','user_score']= 'NaN'
df['user_score'] = df['user_score'].astype(float)#weconverted datatype to float


# tbd is to be determined which we will treat as missing value.

# In[11]:


df['user_score_noempty']= df.groupby('genre')['user_score'].transform(lambda y : y.fillna(y.mean()))
df['user_score_noempty'].isnull().sum()


# In[12]:


df["user_score_noempty"].fillna("0", inplace = True) 
df['user_score_noempty'].isnull().sum()


# We followed same process as above to fill missing values in User Score. User Score has missing values because sometimes user plays a game and user does'nt rate it because he is not fully satisfied and neither fully dissappointed.

# ## 2.3 Calculate the total sales (the sum of sales in all regions) for each game and put these values in a separate column. 

# In[13]:



df['total_sales'] = df['na_sales']+df['eu_sales']+df['jp_sales']+df['other_sales']
df.head()


# We calculated total sales of each game by combining sales in each region

# In[14]:


df.columns


# In[15]:


new_df = df[['name', 'platform', 'genre', 'na_sales', 'eu_sales',
       'jp_sales', 'other_sales','year_of_release_noempty', 'critic_score_noempty',
       'user_score_noempty', 'rating','total_sales']].copy()

new_df = new_df.query('genre != "No genre" and name != "No Name"')


# In[16]:


rating_grouped = new_df.groupby('genre')['rating'].agg(pd.Series.mode).reset_index()
rating_dict=dict(zip(rating_grouped.genre,rating_grouped.rating))
rating_dict


# In[17]:


new_df['rating_noempty'] = new_df['rating'].fillna(new_df['genre'].map(rating_dict))


# Here we have taken all processed data without any missing values.

# In[18]:


new_df.isnull().sum()


# In[19]:


del new_df['rating']


# In[20]:


new_df.head()
new_df.isnull().sum()
new_df.info()


# We have created new dataset which does not have any missing values which will help us in further calculations

# # 3. Analyze the data

# ## 3.1 Look at how many games were released in different years. Is the data for every period significant? 

# In[21]:


games_per_year = new_df.groupby('year_of_release_noempty').agg({'name': ['count']})
games_per_year.head()


# In[22]:


games_per_year.plot(sharex=True, sharey=True,kind='bar',legend = True)
pl.title('games released per year')
pl.xlabel('year')
pl.ylabel('number of games')


# Here we have analysed the data about the number of games released per year and here we can see from year 2000 and further we have more volume of games released per year.
# 
# Number of games released over the period of time but the video games started to become popular from yearly 2000 when high quality gaming become passion for some individuals and then compitition increased in the market along with high quality equipments which comes with platforms as it is directly related to platforms number of platforms released more number of games.
# 

# ## 3.2 Look at how sales varied from platform to platform. Choose the platforms with the greatest total sales and build a distribution based on data for each year. Find platforms that used to be popular but now have zero sales. How long does it generally take for new platforms to appear and old ones to fade? 

# In[23]:


platform_pivot = (new_df.pivot_table(index = ['platform'], values = 'total_sales',aggfunc = 'sum')).reset_index()
platform_pivot_sorted = platform_pivot.sort_values('total_sales',ascending = False)
platform_pivot_sorted.head(5)


# In[24]:


platform_pivot_sorted.plot('platform',sharex=True, sharey=True,kind='bar',legend = True)
pl.title('total sales of games according to platform')
pl.xlabel('platform')
pl.ylabel('number of sales')


# Here we found out popular platform which did great in in total sales.
# when we consider complete data PS2 and X360 are leading platforms.
# 

# ## 3.3 Determine what period you should take data for. To do so, look at your answers to the previous questions. Disregard the data for previous years. 

# In[25]:


year_subset = new_df.query('2014 <= year_of_release_noempty')
year_subset.sort_values('total_sales',ascending = False)


# Here we created new subset with 5 top selling platforms. we have set time period of games released after year 2014 as we don't need previous years because it does not have much volume of games which are still available in market.
# data going back to 2016. Let’s imagine that it’s December 2016 and we’re planning a campaign for 2017.

# In[26]:


platform_sales_pivot = (year_subset.pivot_table(index = ['platform'], values = 'total_sales',aggfunc = 'sum')).reset_index()
platform_sales_pivot_sorted = platform_sales_pivot.sort_values('total_sales',ascending = False)
platform_sales_pivot_sorted.head(5)


# In[27]:


platform_subset = year_subset.query('platform in ("PS4","XOne","3DS","PS3","X360")')
platform_subset.head()


# Our top 5 platforms are: "PS4","XOne","3DS","PS3","X360"

# In[28]:


import sys
import warnings
if not sys.warnoptions:
       warnings.simplefilter("ignore")


# In[29]:


platform_subset["user_score_noempty"] = pd.to_numeric(platform_subset['user_score_noempty'], errors='coerce')
platform_subset.info()


# ## 3.4 Which platforms are leading in sales? Which ones are growing or shrinking? Select several potentially profitable platforms. 

# In[30]:


platform_subset.pivot_table(index="year_of_release_noempty",
                      columns="platform",
                      values="total_sales",
                      aggfunc=np.sum,
                    fill_value=0).plot(kind = 'bar',figsize=(12,7),legend=True)
pl.xlabel('year')
pl.ylabel('number of sales')


# ## 3.5 Build a box plot for the global sales of all games, broken down by platform 

# In[31]:


boxprops = dict(linestyle='-', linewidth=4, color='k')
medianprops = dict(linestyle='-', linewidth=4, color='k')

ax = platform_subset.boxplot(column=['total_sales'],
                by='platform',
                showfliers=False, showmeans=True,
                boxprops=boxprops,
                medianprops=medianprops)
pl.suptitle("")
ax.set_xlabel("platform")
ax.set_title("Boxplot")


# by looking at the boxplot we can say that the leading platforms in sales are PS4 and Xone basically Xbox and PlayStation. 

# In[32]:


boxprops = dict(linestyle='-', linewidth=4, color='k')
medianprops = dict(linestyle='-', linewidth=4, color='k')

ax = platform_subset.boxplot(column=['total_sales'],
                by='platform',
                showfliers=False, showmeans=True,
                boxprops=boxprops,
                medianprops=medianprops)
Q1 = platform_subset['total_sales'].quantile(0.25)
Q3 = platform_subset['total_sales'].quantile(0.75)
IQR = Q3 - Q1
pl.xticks(rotation=90)
pl.hlines(y= [Q1-1.5*IQR], xmin=0.9, xmax=1.1, color='red')
# get rid of the automatic title
pl.suptitle("")
ax.set_xlabel("platform")
ax.set_title("Boxplot")


# PC games are not leading in this chart the reason may be because of quality and user experiance of playing game.

# ## 3.6 Take a look at how user and professional reviews affect sales for one popular platform (you choose). Build a scatter plot and calculate the correlation between reviews and sales.  

# In[33]:


plat_ps4 = platform_subset.query('platform in ("PS4")')
plat_ps4["user_score_noempty"] = pd.to_numeric(plat_ps4['user_score_noempty'], errors='coerce')
plat_ps4.sort_values('total_sales',ascending = False)


# We chose PS4 platform for this task as we have seen in the last task that this platform is highest selling platform for given time period.

# In[34]:


ps4_pivot = (plat_ps4.pivot_table(index = ['name','user_score_noempty'], values = 'total_sales',aggfunc = 'sum')).reset_index()
ps4_pivot = ps4_pivot.sort_values('user_score_noempty',ascending = False)
ps4_pivot.head(5)


# In[35]:


ps4_pivot = ps4_pivot.sort_values('total_sales',ascending = False)
ps4_pivot.head(5)


# As we seen here the top 5 games which are having highest sales in PS4 are: Call of Duty: Black Ops 3	, Grand Theft Auto V,	FIFA 16	, Star Wars Battlefront (2015), Call of Duty: Advanced Warfare	top 5 games which are having highest user scores are in PS4: The Witcher 3: Wild Hunt,Dark Souls III,The King of Fighters XIV,Farming Simulator,Rocket League	
# so, User ratings does not affect sales because user ratings are divided by the choice of genre, visual quality and overall user experiance which differes from platform to platform.

# In[36]:


plat_ps4.plot(x='user_score_noempty', y = 'total_sales', kind = 'scatter', figsize=(8, 6), grid=True, c= 'lightgreen')
pl.title('Scatterplot for platform PS4')


# In[37]:


print(plat_ps4['user_score_noempty'].corr(plat_ps4['total_sales']))


# A correlation of -0.053 implies a connection, though it could be weaker. Increase user_score_noempty and total_sales will often increase, but not always. And vice versa: increase user_score_noempty, and total_sales will often change. Thus, we don't know anything about cause and effect; we only know that the two factors show correlation.
# Thus, you can't prove cause and effect with the presence of correlation, but you won’t disprove it either.

# In[38]:


popular_games_ps4 = plat_ps4["name"]
popular_games_ps4


# ## 3.7 Keeping your conclusions in mind, compare the sales of the same games on other platforms. 

# In[39]:


popular_games_other_plat = platform_subset.query('name in @popular_games_ps4 and platform != "PS4"')
popular_games_other_plat.sort_values('total_sales',ascending = False)


# In[40]:


popular_games_pivot = (popular_games_other_plat.pivot_table(index = ['name','user_score_noempty','platform'], values = 'total_sales',aggfunc = 'sum')).reset_index()
popular_games_pivot = popular_games_pivot.sort_values('total_sales',ascending = False)
popular_games_pivot.head(5)


# In[41]:


popular_games_pivot = popular_games_pivot.sort_values('user_score_noempty',ascending = False)
popular_games_pivot.head(5)


# here almost same games are in top selling 5 games Call of Duty: Black Ops,Grand Theft Auto V ,Minecraft,Call of Duty: Advanced Warfare,Call of Duty: Advanced Warfare	 that could be because games are depend on the platforms. and here also, user score does not depend on top sells.

# In[42]:


popular_games_other_plat["user_score_noempty"] = pd.to_numeric(popular_games_other_plat['user_score_noempty'], errors='coerce')
popular_games_other_plat.plot(x='user_score_noempty', y = 'total_sales', kind = 'scatter', figsize=(8, 6), grid=True, c= 'black')
pl.title('Scatterplot for other platforms')


# In[43]:


print(popular_games_other_plat['user_score_noempty'].corr(popular_games_other_plat['total_sales']))


# A correlation of -0.163 implies a connection, though it could be opposite. Increase user_score_noempty and total_sales will often increase, but not always. And vice versa: increase user_score_noempty, and total_sales will often change. Thus, we don't know anything about cause and effect; we only know that the two factors show correlation.
# Thus, you can't prove cause and effect with the presence of correlation, but you won’t disprove it either.

# In[44]:


boxprops = dict(linestyle='-', linewidth=4, color='k')
medianprops = dict(linestyle='-', linewidth=4, color='k')

ax = popular_games_other_plat.boxplot(column=['total_sales'],
                by='platform',
                showfliers=False, showmeans=True,
                boxprops=boxprops,
                medianprops=medianprops)
Q1 = popular_games_other_plat['total_sales'].quantile(0.25)
Q3 = popular_games_other_plat['total_sales'].quantile(0.75)
IQR = Q3 - Q1
pl.xticks(rotation=90)
pl.hlines(y= [Q1-1.5*IQR], xmin=0.9, xmax=1.1, color='red')
# get rid of the automatic title
pl.suptitle("")
ax.set_xlabel("platform")
ax.set_title("Boxplot")


# box plot sates that top selling platform other than x360 is PS3

# ## 3.8 Take a look at the general distribution of games by genre. What can we say about the most profitable genres? Can you generalize about genres with high and low sales? 

# In[45]:


genre_sales = (year_subset.pivot_table(index = ['genre'], values = 'total_sales',aggfunc = 'sum')).reset_index()
genre_sales_sorted = genre_sales.sort_values('total_sales',ascending = False)
genre_sales_sorted.head()


# In[46]:


genre_sales_sorted.plot('genre',sharex=True, sharey=True,kind='bar')
pl.title('Total sales by Genre')
pl.xlabel('Genre')
pl.ylabel('number of sales')


# Here we found out top 5 genre having top sell: "Action","Shooter","Sports","Role-Playing","Misc"

# In[47]:


genre_subset = year_subset.query('genre in ("Action","Shooter","Sports","Role-Playing","Misc")')
genre_subset.head()


# created subset for top 5 to selling genres

# # 4. Create a user profile for each region

# ## 4.1 The top five platforms. Describe variations in their market shares from region to region. 

# In[48]:


columns_to_clean = ['na_sales','eu_sales','jp_sales','other_sales']
for x in columns_to_clean:
    boxprops = dict(linestyle='-', linewidth=4, color='k')
    medianprops = dict(linestyle='-', linewidth=4, color='k')

    ax = platform_subset.boxplot(column=[x],
                by='platform',
                showfliers=False, showmeans=True,
                boxprops=boxprops,
                medianprops=medianprops)
    Q1 = platform_subset[x].quantile(0.25)
    Q3 = platform_subset[x].quantile(0.75)
    IQR = Q3 - Q1
    pl.xticks(rotation=90)
    pl.hlines(y= [Q1-1.5*IQR], xmin=0.9, xmax=1.1, color='red')
# get rid of the automatic title
    pl.suptitle(" ")
    ax.set_xlabel("Platform")
    ax.set_title(f'{x}')
    


# as we follow boxplot we can say that other than America PS4 has highest sells in  EU and Other regions while, Japan has 3DS platform leading in popularity and NA has Xone.

# In[49]:


for x in columns_to_clean:
    print(platform_subset.groupby('platform').agg({f'{x}': ['mean']}).reset_index())


# ## 4.2  The top five genres. Explain the difference. 

# In[50]:


for x in columns_to_clean:
    print(genre_subset.groupby('genre').agg({f'{x}': ['mean']}).reset_index())


# In[51]:


for x in columns_to_clean:
    boxprops = dict(linestyle='-', linewidth=4, color='k')
    medianprops = dict(linestyle='-', linewidth=4, color='k')

    ax = genre_subset.boxplot(column=[x],
                by='genre',
                showfliers=False, showmeans=True,
                boxprops=boxprops,
                medianprops=medianprops)
    Q1 = genre_subset[x].quantile(0.25)
    Q3 = genre_subset[x].quantile(0.75)
    IQR = Q3 - Q1
    pl.xticks(rotation=90)
    pl.hlines(y= [Q1-1.5*IQR], xmin=0.9, xmax=1.1, color='red')
# get rid of the automatic title
    pl.suptitle(" ")
    ax.set_xlabel("Genre")
    ax.set_title(f'{x}')
    


# selling of games is different in different areas according to genre.
# Except Japan(Role-Playing) in other remaining regions Shooter is popular Genre

# ## 4.3 Do ESRB ratings affect sales in individual regions?  

# In[52]:


for x in columns_to_clean:
    print(year_subset.groupby(['rating_noempty'])[f'{x}'].mean().reset_index())


# In[53]:


for x in columns_to_clean:
    boxprops = dict(linestyle='-', linewidth=4, color='k')
    medianprops = dict(linestyle='-', linewidth=4, color='k')

    ax = year_subset.boxplot(column=[x],
                by='rating_noempty',
                showfliers=False, showmeans=True,
                boxprops=boxprops,
                medianprops=medianprops)
    Q1 = new_df[x].quantile(0.25)
    Q3 = new_df[x].quantile(0.75)
    IQR = Q3 - Q1
    pl.xticks(rotation=90)
    pl.hlines(y= [Q1-1.5*IQR], xmin=0.9, xmax=1.1, color='red')
# get rid of the automatic title
    pl.suptitle(" ")
    ax.set_xlabel("ESRB Rating")
    ax.set_title(f'{x}')
    


# ESRB ratings affect sales in different ways in different areas.
# 1. na_sales = M rating
# 2. eu_sales = M rating
# 3. jp_sales = T rating
# 4. other_sales = M rating

# In[54]:


popular_games_pivot = (year_subset.pivot_table(index = ['name','rating_noempty','year_of_release_noempty'], values = 'total_sales',aggfunc = 'sum')).reset_index()
popular_games_pivot = popular_games_pivot.sort_values('total_sales',ascending = False)
popular_games_pivot.head(5)


# here we can see  that the highest grossing games on different platforms are different and our top 5 grossing games are have M rating.

# In[55]:


new_df['platform'].unique()


# # 5. Test the following hypotheses

# In[56]:


x_one_data = year_subset.query('platform in ("XOne")')
x_one_data.columns


# In[57]:


pc_data = year_subset.query('platform in ("PC")')


# we have created two separate dataset to run hypotheses

# ## 5.1 Average user ratings of the Xbox One and PC platforms are the same 

# In[58]:


from scipy.stats import mannwhitneyu


# In[59]:


print(x_one_data['user_score_noempty'].mean())
variance = np.var(x_one_data['user_score_noempty'])
print(variance)
standard_deviation = np.sqrt(variance)
print(standard_deviation)


# In[60]:


print(pc_data['user_score_noempty'].mean())
variance = np.var(pc_data['user_score_noempty'])
print(variance)
standard_deviation = np.sqrt(variance)
print(standard_deviation)


# In[61]:


stat, p = mannwhitneyu(x_one_data['user_score_noempty'], pc_data['user_score_noempty'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
 print('Same distribution (fail to reject H0)')
else:
 print('Different distribution (reject H0)')


# The Mann-Whitney U test is a nonparametric statistical significance test for determining whether two independent samples were drawn from a population with the same distribution.
# The default assumption or null hypothesis is that there is no difference between the distributions of the data samples. Rejection of this hypothesis suggests that there is likely some difference between the samples. More specifically, the test determines whether it is equally likely that any randomly selected observation from one sample will be greater or less than a sample in the other distribution. If violated, it suggests differing distributions.
# 
# Fail to Reject H0: Sample distributions are equal.
# Reject H0: Sample distributions are not equal.
# 
# Running the example calculates the test on the datasets and prints the statistic and p-value.
# The p-value strongly suggests that the sample distributions are different, means Average user ratings of the Xbox One and PC platforms are not the same.
# 

# ## 5.2 Average user ratings for the Action and Sports genres are different 

# In[62]:


new_df['genre'].unique()


# In[63]:


action_data = year_subset.query('genre in ("Action")')


# In[64]:


print(action_data['user_score_noempty'].mean())
variance = np.var(action_data['user_score_noempty'])
print(variance)
standard_deviation = np.sqrt(variance)
print(standard_deviation)


# In[65]:


sports_data = year_subset.query('genre in ("Sports")')


# In[66]:


print(sports_data['user_score_noempty'].mean())
variance = np.var(sports_data['user_score_noempty'])
print(variance)
standard_deviation = np.sqrt(variance)
print(standard_deviation)


# In[67]:


stat, p = mannwhitneyu(action_data['user_score_noempty'], sports_data['user_score_noempty'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
 print('Same distribution (fail to reject H0)')
else:
 print('Different distribution (reject H0)')


# Running the example calculates the test on the datasets and prints the statistic and p-value.
# The p-value strongly suggests that the sample distributions are different, means AAverage user ratings for the Action and Sports genres are different.
# Level of significance alpha: Refers to the degree of significance in which we accept or reject the null-hypothesis. 100% accuracy is not possible for accepting or rejecting a hypothesis, so we therefore select a level of significance that is usually 5%.
# This is generally it is 0.05 or 5% , which means your output should be 95% confident to give similar kind of result in each sample.

# # Final Conclusion

# The aim of the project was 'You need to identify patterns that determine whether a game succeeds or not. This will allow you to spot potential big winners and plan advertising campaigns'
# My findings are:
# 1. Top 5 popular platforms: "PS4","XOne","3DS","PS3","X360"(Leadin two are:PS4 and Xone)
# 2. Top 5 popular genres: "Action","Shooter","Sports","Role-Playing","Misc"(Leading are: Action, Shooter and Role-Playing in Japan)
# 3. Top 5 popular games common for all above: Call of Duty: Black Ops 3	, Grand Theft Auto V,	FIFA 16	, Star Wars Battlefront (2015), Call of Duty: Advanced Warfare,Minecraft
# 4. Top ESRB rating for this games is: M and T for Japan
# 5. Considered period for calculation : 2014 and above
# 
# by analysing all the given data we have to focus on above parameters for our advertising campaign.

# In[ ]:




