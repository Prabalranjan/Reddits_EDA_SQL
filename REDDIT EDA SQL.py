#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This is a brief exploratory data analysis using Pandas for a given public sample of random Reddit posts. We will get a feel of a dataset and try to answer the following questions:

# What are the most popular reddits? Which topics are viral?
# Which posts have been removed and why?
# What % removed reddits are deleted by moderatos?
# Who are the most popular authors?
# Who are the biggest spammers at Reddit platform?


# In[7]:


#getting all the packages
import numpy as np #linear algebra
import pandas as pd #data processing

import seaborn as sns #satistics graph package
import matplotlib.pyplot as plt # plot package
import pandasql as ps # sql package
import wordcloud # will user for world cloud plot
from wordcloud import WordCloud, STOPWORDS # optional to filter out the stopwords

#optional helpful plot stypes:
plt.style.use('bmh') #setting up 'bmh' as "Bayesian Methods for Hackers" style sheet
#plt.style.use('ggplot') #R ggplot stype
#print(plt.style.available) #pick another style


# # Reading the dataset

# In[43]:


df = pd.read_csv("r_dataisbeautiful_posts.csv")


# In[13]:


# df.head()
df.sample(4)


# In[15]:


df.tail()


# In[19]:


print("Data shape:", df.shape)  #to see the shape total row and column.


# # Getting a feel of the dataset
# Basic EDA commands

# In[21]:


df.info()


# In[29]:


df.describe()  # gives the numerrical details


# In[31]:


df.isnull() #givves false if it is not null
df.isnull().sum() #gives the summ of all the isnull() according to column.


# In[37]:


df.isnull().sum().sort_values(ascending= False) #sort_values(ascending= FALSE) sort the value in descending order.


# # Removed reddits deep dive
# Let's see who and why removes posts:

# In[54]:


q1= """SELECT removed_by, count(DISTINCT id) AS number_of_removed_post
FROM df
WHERE removed_by IS NOT NULL
GROUP BY removed_by"""

q2 = """SELECT removed_by
FROM df"""

# print(q1)
grouped_df = ps.sqldf(q1,locals())
grouped_df


# In[60]:


# visualize bar chart based of sql output:

removed_by = grouped_df['removed_by'].tolist()

number_of_removed_post = grouped_df['number_of_removed_post'].tolist()

plt.figure(figsize=(12,8))
plt.ylabel("Number of deleted reddits")
plt.bar(removed_by,number_of_removed_post)

plt.show()


# ### Who are the top 3 users who had the most their posts removed by moderator?

# In[66]:


q3 = """SELECT author, count(removed_by) as no_of_removed_post
FROM df
WHERE removed_by = 'moderator'
group by author
order by no_of_removed_post DESC
LIMIT 5"""

authDEL = ps.sqldf(q3,locals())
authDEL


# ## Let's find out how many posts with "virus" keyword are removed by moderator.

# In[74]:


q4 = """SELECT count(removed_by)
FROM df
WHERE removed_by = 'moderator'
AND title LIKE '%virus%'"""

# print(ps.sqldf(q4,locals()))

removed_moderator_virus = ps.sqldf(q4,locals())
print(removed_moderator_virus.values[0])


# ### getting % virus reddits from all removed posts:
# 

# In[82]:


q5 = """SELECT count(removed_by)
FROM df
WHERE removed_by LIKE '%moderator%'"""

all_removed_moderator = ps.sqldf(q5,locals())
all_removed_moderator


# In[84]:


print(removed_moderator_virus/all_removed_moderator)


# From all removed reddits by moderator, 11.37% ~ 12% posts contain the "virus" keyword.

# ### The most popular reddits
# Top 10 reddits with the most number of comments:
# 

# In[92]:


q6 = """
SELECT id, title, num_comments
FROM df
WHERE title <> "data_irl"
ORDER BY num_comments DESC
LIMIT 10;
"""

print(ps.sqldf(q6,locals()))


# ## The most common words in reddits:
# Let's see the word map of the most commonly used words from reddit titles:

# In[96]:


# to build a worldcloud , we have to remove NULL values first:
df["title"] = df["title"].fillna(value="")


# In[98]:


#Now let's add a string value instead to make our Series clean:
word_string = " ".join(df["title"].str.lower())


# In[100]:


#And - plotting:

plt.figure(figsize=(15,15))
wc = WordCloud(background_color="purple", stopwords = STOPWORDS, max_words=2000, max_font_size= 300,  width=1600, height=800)
wc.generate(word_string)

plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), interpolation="bilinear")
plt.axis('off')


# ## Comment Distribution
# The average reddit has less than 25 comments. Let's see the comment distribution for those reddits who have <25 comments:

# In[102]:


fig, ax = plt.subplots()
_ = sns.distplot(df[df["num_comments"] < 25]["num_comments"], kde=False, rug=False, hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="num_comments", ylabel="id")

plt.ylabel("Number of reddits")
plt.xlabel("Comments")

plt.show()


# ## Correlation between dataset variables
# Now let's see how the dataset variables are correlated with each other:
# 
# 1. How score and comments are correlated?
# 2. Do they increase and decrease together (positive correlation)?
# 3. Does one of them increase when the other decrease and vice versa (negative correlation)? Or are they not correlated?
# 
# Correlation is represented as a value between -1 and +1 where +1 denotes the highest positive correlation, -1 denotes the highest negative correlation, and 0 denotes that there is no correlation.
# 
# Let's see the correlation table between our dataset variables (numerical and boolean variables only)

# In[104]:


df.corr()


# In[106]:


# Now let's visualize the correlation table above using a heatmap
h_labels = [x.replace('_', ' ').title() for x in 
            list(df.select_dtypes(include=['number', 'bool']).columns.values)]

fig, ax = plt.subplots(figsize=(10,6))
_ = sns.heatmap(df.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)


# ## Score Distribution

# In[108]:


df.score.describe()


# In[110]:


df.score.median()


# In[119]:


fig, ax = plt.subplots()
_ = sns.distplot(df[df["score"] < 20]["score"], kde=False, hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="score", ylabel="No. of reddits")

