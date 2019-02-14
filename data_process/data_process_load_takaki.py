import numpy as np
import pandas as pd

#%%
# load all data
df_people=pd.read_csv('dataset/org/people.csv')
df_act_train=pd.read_csv('dataset/org/act_train.csv')
df_act_test=pd.read_csv('dataset/org/act_test.csv')

# rename key name to avoid depulication
for i in range(1,39):
    keyname_old='char_'+str(i)
    keyname_new='p'+keyname_old
    df_people=df_people.rename(columns={keyname_old: keyname_new})

# merge 2 dataset
df_merge_train=pd.merge(df_people,df_act_train,on='people_id',how='right') # remain all values in df_act_train (right)

# %%
# count number of data type
type_num_of_activity_category=df_act_train['activity_category'].value_counts() # 7 types (type1 - type7)
print(type_num_of_activity_category)


#%%
# for reducing calculation for prototyping, only use 10000 samples
df_merge_train_10000samples=df_merge_train[0:10000]

# delete char_1 to char_9 because it seems complex right now.
for i in range(1,10):
    deleted_key='char_'+str(i)
    del df_merge_train_10000samples[deleted_key]

# get_one_hot_expression
df_one_hot_simple=pd.get_dummies(df_merge_train_10000samples,columns=['activity_category'])

df_one_hot_simple.to_csv('dataset_test.csv')
