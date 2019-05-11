import pandas as pd
import matplotlib.pylab as plt
from matplotlib import pyplot
import numpy as np

# Lab 1
other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(other_path, header=None)

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df.columns = headers    


df.dropna(subset=["price"], axis=0)  

df.to_csv("automobile.csv", index=False)

#print(df.dtypes)
#print(df.describe())
#print(df.describe(include = "all"))
#print(df.info)

#Lab 2

df.replace("?", np.nan, inplace = True)

#print(df.head(5))

missing_data = df.isnull()   ## Detects Missing Data
#print(missing_data.head(5))

##This loop counts missing values in each column

#for column in missing_data.columns.values.tolist():  
#    print(column)
#    print (missing_data[column].value_counts())   
#    print("")    

 #calculated avg of the column    
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)   
#print("Average of normalized-losses:", avg_norm_loss)   

#replaced NaN with avg of the column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)    

avg_bore=df['bore'].astype('float').mean(axis=0)
#print("Average of bore:", avg_bore)

df["bore"].replace(np.nan, avg_bore, inplace = True)

avg_stroke=df['stroke'].astype('float').mean(axis=0)
#print("Average of stroke", avg_stroke)

df["stroke"].replace(np.nan,avg_stroke,inplace=True)

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
#print("Average horsepower:", avg_horsepower)

df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
#print("Average peak rpm:", avg_peakrpm)

df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

#print(df['num-of-doors'].value_counts())
#print(df['num-of-doors'].value_counts().idxmax()) #idxmax calculates most common type

#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

#correcting data types
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

df["highway-l/100km"]= 235/df["highway-mpg"]

# replace (original value) by (original value)/(maximum value)
#for data normalization
df['length'] = df['length']/df['length'].max()
df["height"] = df["height"]/df["height"].max()
#print(df[["length","width","height"]].head())

#binning

df["horsepower"]=df["horsepower"].astype(int, copy=True)


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
#print(bins)
df['width'] = df['width']/df['width'].max()
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
#print(df[['horsepower','horsepower-binned']].head(20))

#pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
#plt.pyplot.xlabel("horsepower")
#plt.pyplot.ylabel("count")
#plt.pyplot.title("horsepower bins")

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
#print(dummy_variable_1.head())

dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
#print(dummy_variable_1.head())

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

#print(df.head())

dummy2 = pd.get_dummies(df["aspiration"])
dummy2.rename(columns = {"std":"aspiration-std","turbo":"aspiration turbo"},inplace=True)
#print(dummy2.head())

df = pd.concat([df,dummy2],axis=1)
df.drop("aspiration",axis=1,inplace=True)
print(df.head())

df.to_csv('automobile1.csv')

#Lab 3
