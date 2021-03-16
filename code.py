import pandas as pd
df1=pd.read_csv("C:/Users/thiyaneswaranm/Downloads/transcripts.csv")
# print(df1["Remarks_data"])
df10,df11 = pd.DataFrame(columns=['A', 'B', 'M']),pd.DataFrame(columns=['A', 'B', 'M'])
df1.dropna(subset=["Remarks"], inplace=True)
#print(df1.head(5))
#df1.dropna("Suugestion1",inplace=True)
# print(df1)
# idx=df1[df1["Remarks"], ["confidence"]==0].index
# print(idx)
# df1.drop(idx,inplace=True)
# df1 = df1.reset_index(drop=True)
#df1.reset_index(drop=True)
# print(df1)
#df = pd.DataFrame.from_dict(d, orient='index')
for i in range(1,3):
    dfr=df1["Remarks"][i]
    file1 = open("myfile.json","w")
    # print (i)
#     print (dfr)
    file1.write(str(dfr))
    file1.close()
    # import pandas as pd
    df=pd.read_json("myfile.json")
#     print (df)
    df=df.T
    index_list = [0]
    dff=df.take(index_list)
    df10=df10.append(dff)
    index_list = [1]
    dfc=df.take(index_list)
    df11=df11.append(dfc)
# print(df10)
# print(df11)
df10.to_csv("test_catch1.csv")
df10=pd.read_csv("test_catch1.csv")
df10=df10[['0','1','2']]
df10.rename(columns={"0": "int0", "1": "int1","2":"int2"},inplace=True)
print(df10)
#add=df1.join(df10)
#add.head()
# add.to_csv("add.csv")
df11.to_csv("test_catch2.csv")
df11=pd.read_csv("test_catch2.csv")
df11=df11[['0','1','2']]
df11.rename(columns={"0": "conf0", "1": "conf1","2":"conf2"},inplace=True)
print(df11)
# add=add.join(df10,df11)
# dfs = [df10, df11]
# dfs = [df.set_index('intent') for df1 in dfs]
# dfs[0].join(dfs[1:])

# add.head()
#add.to_csv("add.csv")

dat=df10.join(df11)
dat
pd.concat([df1, dat], axis=1)
# dfs = [df1, df10, df11]
# # dfs = [df for df in dfs]
# dfs[0].join(dfs[1:])