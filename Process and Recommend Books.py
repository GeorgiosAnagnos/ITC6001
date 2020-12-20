#this may be needed to import the isbn module the first time. Then run import below 
import isbnlib
import mysql.connector
import pandas as pd
import scipy.stats as stats
import numpy as np
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt
import math
import json

#define all the functions to be used for the recommendation
def csim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
 
# Pearson similarity of users u and v
def psim(u, v):
    return csim(u - np.mean(u), v - np.mean(v))
 
def calc_similarities(r, sim=csim):
    return [[sim(u1, u2) for u1 in r] for u2 in r]

#calculate the k neighbors for user s based on rating similarity proximity
def calc_neighbourhood(s, k):
    return [[x for x in np.argsort(s[i]) if x != i][len(s) - 1: len(s) - k - 2: -1] for i in range(len(s))]

def validate_and_transform_to_isbn_13(df,column_name):
    df[column_name] = df[column_name].apply(check_and_transform_to_isbn13)
    records_to_drop = df[df[column_name] == -1].index
    df.drop(records_to_drop, inplace=True)
    print(records_to_drop)
    print('dropped' ,len(records_to_drop), 'due to invalid isbn')
    return df

def check_and_transform_to_isbn13(isbn):
    if(isbnlib.is_isbn10(isbn)):
        return str(isbnlib.to_isbn13(isbn))
    if(isbnlib.is_isbn13(isbn)):
        return str(isbn)
    return str(isbn)

#removes data from column_name based on a z score threshold
def remove_outliers_based_on_z_score(df,column_name,z_score_threshold):
    cleaned_df = df.copy()
    cleaned_df[column_name + ' count'] = df.groupby(column_name)[column_name].transform('count')
    zscores =  np.abs(stats.zscore(cleaned_df[column_name + ' count']))
    records_to_drop = cleaned_df[(zscores > z_score_threshold)].index
    cleaned_df.drop(records_to_drop, inplace=True)
    print('dropped' ,len(records_to_drop), 'records using column',column_name )
    return cleaned_df

#predicts user ratings for each Item based on nearest neighbor
def predict(userId, itemId, r, s, nb):
    rsum, ssum = 0.0, 0.0
    for n in nb[userId]:
        rsum += s[userId][n] * (r[n][itemId] - np.mean(r[n]))
        ssum += s[userId][n]
    return np.mean(r[userId]) + rsum / ssum

# mae(p, a) returns the mean average error between
def mae(p, a):
    return sum(map(lambda x: abs(x[0] - x[1]), zip(p, a))) / len(p)


# rmse(p, a) returns the root mean square error between
def rmse(p, a):
    return math.sqrt(sum(map(lambda x: (x[0] - x[1]) ** 2, zip(p, a))) / len(p))

# flatten(l) flattens a list of lists l
def flatten(l):
    return [x for r in l for x in r]

#%% Set up the dataframes from the SQL database

db = mysql.connector.connect(user='root', password='123123', host='127.0.0.1')
cursor=db.cursor()
cursor.execute("use books")
#db = pymysql.connect(host="localhost",
#                     user="root",
#                     passwd="123123")
 ###populate database
booksDf = pd.read_sql("select * from books.`BX-Books`", db)
print('bookDuplicates = ', booksDf.duplicated(['ISBN']).sum())

usersDf = pd.read_sql("select * from books.`BX-Users`", db)
print('userDuplicates = ', usersDf.duplicated(['User-ID']).sum())

ratingsDf = pd.read_sql("select * from books.`bx-book-ratings`", db)
print('ratingDuplicates = ', ratingsDf.duplicated(['User-ID','ISBN']).sum())
#create dataframe keeping only books that have ratings
explicitRatingsDf = pd.read_sql("select * from books.`bx-book-ratings` where `Book-Rating` > 0 ", db)

#%%

# change all ISBN10 to ISBN13 and drop duplicates
booksDf = validate_and_transform_to_isbn_13(booksDf,'ISBN')
print('bookDuplicates = ', booksDf.duplicated(['ISBN']).sum())
booksDf.drop_duplicates(subset ='ISBN',keep ='first', inplace = True)
ratingsDf = validate_and_transform_to_isbn_13(ratingsDf,'ISBN')
print('ratingDuplicates = ', ratingsDf.duplicated(['User-ID','ISBN']).sum())
ratingsDf.drop_duplicates(subset =['ISBN','User-ID'],keep ='first', inplace = True)
explicitRatingsDf = validate_and_transform_to_isbn_13(explicitRatingsDf,'ISBN')
print('explicitRatingDuplicates = ', explicitRatingsDf.duplicated(['User-ID','ISBN']).sum())
explicitRatingsDf.drop_duplicates(subset =['ISBN','User-ID'],keep ='first', inplace = True)
print('bookDuplicates = ', booksDf.duplicated(['ISBN']).sum())
print('ratingDuplicates = ', ratingsDf.duplicated(['User-ID','ISBN']).sum())
print('explicitRatingDuplicates = ', explicitRatingsDf.duplicated(['User-ID','ISBN']).sum())


"""
this cleans up ratings df based on z score of User-ID
"""
cleanedRatingsDf= remove_outliers_based_on_z_score(explicitRatingsDf,'User-ID',3)
cleanedRatingsDf= remove_outliers_based_on_z_score(explicitRatingsDf,'ISBN',3)

##merge dataframes
ratingsWithBooksDf = pd.merge(cleanedRatingsDf,booksDf, on='ISBN', how='inner')
cleanedRatingsWithBooksDf = remove_outliers_based_on_z_score(ratingsWithBooksDf,'Book-Author',3)
completeDf = pd.merge(cleanedRatingsWithBooksDf,usersDf, on = 'User-ID', how='inner')
completeDf.info(memory_usage='deep')

#completeDf.to_csv('BooksMerged.csv')
minimum_info_complete_Df = completeDf[['ISBN', 'User-ID', 'Book-Rating']]
minimum_info_complete_Df.info(memory_usage='deep')

#Display in reverse order (least popular first) the book popularity, author popularity, and users by reading activity
pd.set_option('display.max_columns', None)

SortedBooksDf = completeDf.sort_values(by=['ISBN count'], ascending=False)
SortedBooksDf.head(10)
SortedAuthorsDf = completeDf.sort_values(by=['Book-Author count'], ascending=False)
SortedAuthorsDf.head(10)

#count and sort users
completeDf2 = completeDf
completeDf2['UserCounts'] = completeDf.groupby('User-ID')['User-ID'].transform('count')
SortedAgesDf = completeDf2.sort_values(by=['UserCounts', 'Age'], ascending=False)
SortedAgesDf.head(10)

#check means for authors
completeDf['Book-Author count'].median()
completeDf['Book-Author count'].mean()



#%%
small_minimum_info_complete_Df = minimum_info_complete_Df[:8000].copy()
small_minimum_info_complete_Df['BookRating'] = minimum_info_complete_Df['Book-Rating'].astype(np.uint8)
df = small_minimum_info_complete_Df
user_ids = list(sorted(df['User-ID'].unique()))
book_ids = list(sorted(df['ISBN'].unique()))
data = df['BookRating'].tolist()
row = df['User-ID'].astype('category').cat.codes
col = df['ISBN'].astype('category').cat.codes
encoded_user_ids = dict( zip( df['User-ID'].astype('category').cat.codes, df['User-ID'] ) )
encoded_ISBNS = dict( zip( df['ISBN'].astype('category').cat.codes, df['ISBN'] ) )

sparse_matrix = sparse.csr_matrix((data, (row, col)), shape=(len(user_ids), len(book_ids)))

#%%
#calculate user similarities and save to CSV
s = calc_similarities(sparse_matrix.toarray())
user_similarity = pd.DataFrame(s,columns=user_ids, index=user_ids)
user_similarity.to_csv('user-pairs-books.data')

#create relational database with user-pairs
create_table_user_pairs = """
CREATE TABLE `user-pairs` (
  `Pair-ID` int(20) NOT NULL default '0',
  `User-ID` int(11) NOT NULL default '0',
  `User-ID-2` int(11) NOT NULL default '0',
  `Similarity` NUMERIC(5,4) NOT NULL default '0',
  PRIMARY KEY  (`Pair-ID`)
) ENGINE=MyISAM;
"""
cursor.execute("drop table if exists `user-pairs`")
cursor.execute(create_table_user_pairs)
counter = 0;

#populate database with user similarities from the user_similarity dataframe
clm = list(user_similarity.columns)
rows=clm.copy()
sqls = []
for column in clm:
    for row in rows:
        similarity=user_similarity[row][column]
        if (row!= column):
            counter +=1
            sql = "INSERT INTO `user-pairs` ( `Pair-ID`,`User-ID`,`User-ID-2`,`Similarity`) VALUES ("+str(counter)+","+str(row)+","+str(column)+","+"%.4f" % similarity+")"
            sqls.append(sql)
            cursor.execute(sql)
db.commit()

#%%
#Calculate the neighborhood
nb = calc_neighbourhood(s, 2)
neighborhoods = {}

for neighborhood in range(0,len(nb)):
    print(nb[neighborhood])   
    new_neighborhood = []
    for neighbor in nb[neighborhood]:
        new_neighbor = clm[neighbor]
        new_neighborhood.append(new_neighbor)
    print(neighborhood)   
    neighborhoods[clm[neighborhood]] = new_neighborhood
    
#store nearest neighbor as json
with open('neighbors-k-books.data', 'w') as fout:
    json.dump(neighborhoods,fout,indent=4)

#%%
#Store neighbors table in sql database

create_table_user_neighborhoods = """
CREATE TABLE `user-neighbors` (
  `User-ID` int(11) NOT NULL default '0',
  `Neighbors` varchar(3000) default NULL,
  PRIMARY KEY  (`User-ID`)
) ENGINE=MyISAM;
"""

cursor.execute("drop table if exists `user-neighbors`")
cursor.execute(create_table_user_neighborhoods)

#populate database
for item in neighborhoods.keys():
    sql = "INSERT INTO `user-neighbors` ( `User-ID`,`Neighbors`) VALUES ("+str(item)+",'"+str(neighborhoods[item])+"')"
    sqls.append(sql)
    cursor.execute(sql)

#%%
#predict recommendations based on neighbor
pr = [[predict(u, i, sparse_matrix.toarray(), s, nb) for i in range(len(sparse_matrix.toarray()[u]))] for u in range(len(sparse_matrix.toarray()))]
predictionDf = pd.DataFrame(pr)
predictionDf.to_csv('predictions.data')

#Calculate MAE and RMSE
print('\nMAE: {:.4f}'.format(mae(flatten(sparse_matrix.toarray()), flatten(pr))))
print('\nRMSE: {:.4f}'.format(rmse(flatten(sparse_matrix.toarray()), flatten(pr))))

#%%
#Find each user's highest predicted rating item and return as a list
top_predictions = []
for u in pr:
    top_prediction = u.index(max(u))
    top_predictions.append(top_prediction)


print(top_predictions)