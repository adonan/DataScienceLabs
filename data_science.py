import pandas as pd
import matplotlib.pylab as plt
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =============================================================================
# =============================================================================
# # ##### Lab 1 #####
# =============================================================================
# =============================================================================
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

# =============================================================================
# =============================================================================
# # ##### Lab 2 #####
# =============================================================================
# =============================================================================

df.replace("?", np.nan, inplace = True)

#print(df.head(5))

# =============================================================================
# ## Detects Missing Data
# =============================================================================
missing_data = df.isnull()   
#print(missing_data.head(5))

# =============================================================================
# ##This loop counts missing values in each column
# =============================================================================

#for column in missing_data.columns.values.tolist():  
#    print(column)
#    print (missing_data[column].value_counts())   
#    print("")    

# =============================================================================
#  #calculated avg of the column    
# =============================================================================
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)   
#print("Average of normalized-losses:", avg_norm_loss)   

# =============================================================================
# #replaced NaN with avg of the column
# =============================================================================
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

# =============================================================================
# #idxmax calculates most common type
# =============================================================================
#print(df['num-of-doors'].value_counts())
#print(df['num-of-doors'].value_counts().idxmax()) 

# =============================================================================
# #replace the missing 'num-of-doors' values by the most frequent 
# =============================================================================
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# =============================================================================
# # simply drop whole row with NaN in "price" column
# =============================================================================
df.dropna(subset=["price"], axis=0, inplace=True)

# =============================================================================
# # reset index, because we droped two rows
# =============================================================================
df.reset_index(drop=True, inplace=True)

# =============================================================================
# #correcting data types
# =============================================================================
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# =============================================================================
# # Convert mpg to L/100km by mathematical operation (235 divided by mpg)
# =============================================================================
df['city-L/100km'] = 235/df["city-mpg"]

df["highway-l/100km"]= 235/df["highway-mpg"]

# =============================================================================
# # replace (original value) by (original value)/(maximum value)
# #for data normalization
# =============================================================================
df['length'] = df['length']/df['length'].max()
df["height"] = df["height"]/df["height"].max()
#print(df[["length","width","height"]].head())

# =============================================================================
# =============================================================================
# # ##### binning #####
# =============================================================================
# =============================================================================

df["horsepower"]=df["horsepower"].astype(int, copy=True)


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
#print(bins)
df['width'] = df['width']/df['width'].max()
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
#print(df[['horsepower','horsepower-binned']].head(20))

#pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# =============================================================================
# # set x/y labels and plot title
# =============================================================================
#plt.pyplot.xlabel("horsepower")
#plt.pyplot.ylabel("count")
#plt.pyplot.title("horsepower bins")

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
#print(dummy_variable_1.head())

dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
#print(dummy_variable_1.head())

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# =============================================================================
# # drop original column "fuel-type" from "df"
# =============================================================================
df.drop("fuel-type", axis = 1, inplace=True)

#print(df.head())

dummy2 = pd.get_dummies(df["aspiration"])
dummy2.rename(columns = {"std":"aspiration-std","turbo":"aspiration turbo"},inplace=True)
#print(dummy2.head())

df = pd.concat([df,dummy2],axis=1)
df.drop("aspiration",axis=1,inplace=True)
#print(df.head())

df.to_csv('automobile1.csv')

# =============================================================================
# =============================================================================
# # #####Lab 3#####
# =============================================================================
# =============================================================================

#print(df.corr())

#print(df[['bore','stroke' ,'compression-ratio','horsepower']].corr())


# =============================================================================
# # Engine size as potential predictor variable of price
# =============================================================================
#sns.regplot(x="engine-size", y="price", data=df)
#plt.ylim(0,)

#print(df[["engine-size", "price"]].corr())

#sns.regplot(x="highway-mpg", y="price", data=df)

#sns.regplot(x="peak-rpm", y="price", data=df)

#sns.boxplot(x="body-style", y="price", data=df)

#sns.boxplot(x="engine-location", y="price", data=df)

#sns.boxplot(x="drive-wheels", y="price", data=df)

#print(df.describe(include=['object']))

# =============================================================================
# #Value Counts
# =============================================================================

#print(df['drive-wheels'].value_counts())
#print(df['drive-wheels'].value_counts().to_frame())

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
#print(drive_wheels_counts)

drive_wheels_counts.index.name = 'drive-wheels'
#print(drive_wheels_counts)

engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
#print(engine_loc_counts.head(10))


# =============================================================================
# =============================================================================
# # #####Groupby#####
# =============================================================================
# =============================================================================

#print(df['drive-wheels'].unique())

df_group_one = df[['drive-wheels','body-style','price']]
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
#print(df_group_one)

# =============================================================================
# # grouping results
# =============================================================================
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
#print(grouped_test1)

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
#print(grouped_pivot)

grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
#print(grouped_pivot)

# =============================================================================
# # grouping results
# =============================================================================
df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
#print(grouped_test_bodystyle)

#plt.pcolor(grouped_pivot, cmap='RdBu')
#plt.colorbar()
#plt.show()

#fig, ax = plt.subplots()
#im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
#row_labels = grouped_pivot.columns.levels[1]
#col_labels = grouped_pivot.index

#move ticks and labels to the center
#ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
#ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
#ax.set_xticklabels(row_labels, minor=False)
#ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
#plt.xticks(rotation=90)

#fig.colorbar(im)
#plt.show()

#p-value is  <  0.001:there is strong evidence that the correlation is significant.
#p-value is  <  0.05: there is moderate evidence that the correlation is significant.
#p-value is  <  0.1: there is weak evidence that the correlation is significant.
#p-value is  >  0.1: there is no evidence that the correlation is significant.

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# =============================================================================
# Since the p-value is  <  0.001, the correlation between wheel-base and price is statistically 
# significant, although the linear relationship isn't extremely strong (~0.585)
# =============================================================================

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# =============================================================================
# Since the p-value is  <  0.001, the correlation between horsepower and price is statistically 
# significant, and the linear relationship is quite strong (~0.809, close to 1)
# =============================================================================

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 

# =============================================================================
# Since the p-value is  <  0.001, the correlation between length and price is statistically 
# significant, and the linear relationship is moderately strong (~0.691).
# =============================================================================

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 

# =============================================================================
# Since the p-value is < 0.001, the correlation between width and price is statistically 
# significant, and the linear relationship is quite strong (~0.751).
# =============================================================================

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
#print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value ) 

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
#print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value )

# =============================================================================
# =============================================================================
# # ##### ANOVA = Analysis of Variance #####
# =============================================================================
# =============================================================================

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
#print(grouped_test2.head(2))
#print(grouped_test2.get_group('4wd')['price'])



f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
#print( "ANOVA results: F=", f_val, ", P =", p_val) 

# =============================================================================
# #fwd and rwd  
# =============================================================================

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
#print( "ANOVA results: F=", f_val, ", P =", p_val )

# =============================================================================
# #4wd and rwd
# =============================================================================
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
#print( "ANOVA results: F=", f_val, ", P =", p_val) 

# =============================================================================
# #4wd and fwd
# =============================================================================
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
#print("ANOVA results: F=", f_val, ", P =", p_val)   


# =============================================================================
# =============================================================================
# # ##### Conclusion #####
# =============================================================================
# =============================================================================

# =============================================================================
# We now have a better idea of what our data looks like and 
# which variables are important to take into account when predicting the car price.
# We have narrowed it down to the following variables:
# 
# Continuous numerical variables:
# 
# Length
# Width
# Curb-weight
# Engine-size
# Horsepower
# City-mpg
# Highway-mpg
# Wheel-base
# Bore
# Categorical variables:
# 
# Drive-wheels
# As we now move into building machine learning models to automate our analysis, 
# feeding the model with variables that meaningfully affect our target variable 
# will improve our model's prediction performance.
# =============================================================================
