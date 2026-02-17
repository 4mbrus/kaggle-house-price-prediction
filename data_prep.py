#%% Importing libraries
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
#%% Data exploration
df = pd.read_csv("data/train.csv")
df.shape
df.info()
# %%
display(df.head(5))
display(df.isna().sum())
#LotFrontage NA = NAN doesnt make much sense, impute with median based on neighbourhood since it is a continuous variable and there are 259 missing values
#Alley NA = No alley access (categorical variable)
#MasVnrType NA = No masonry veneer (categorical variable)
#MasVnrArea NA = ? impute 0 since there is 871 missing in MasVnrType and here only 861 (float variable)
#Bsm... NA = No basement (categorical variables)
#FireplaceQu NA = No fireplace (categorical variable)
#GarageType NA = No garage (categorical variable)
#GarageYrBlt NA = No garage (int) Use YearBuilt for imputation
#GarageFinish NA = No garage (categorical variable)
#GarageQual NA = No garage (categorical variable)
#GarageCond NA = No garage (categorical variable)
#PoolQC NA = No pool (categorical variable)
#Fence NA = No fence (categorical variable)
#MiscFeature NA = No misc feature (categorical variable)
#Electrical NA = ? impute most common value (categorical variable)
# %%
#The MSSubClass is a categorical variable but it is encoded as an integer, we can convert it to a string to treat it as a categorical variable
df["MSSubClass"] = df["MSSubClass"].astype(str)
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df["Electrical"].value_counts() # most common is SBrkr, so we can impute that for the missing value
df["Electrical"].fillna("SBrkr", inplace=True)
df["MasVnrArea"].fillna(0, inplace=True)
df["GarageYrBlt"].fillna(df["YearBuilt"], inplace=True)
# For the rest of the categorical variables, we can impute with "None"
columns_to_impute_none = [
    "Alley",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]
for col in columns_to_impute_none:
    df[col].fillna("None", inplace=True)

#Checking if there are any more missing values
df.isna().sum()

# %% Since our train data is only 1460 rows, we dont want to convert all the categorical variables into dummies, 
# let's see value counts for each categorical variable and decide which ones to convert into dummies
# Lets aim for a maximum of 150 features
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    print(f"{col} value counts:")
    print(df[col].value_counts())
    print("\n")
# %%
#Convert "Street" to dummy variable since it has only 2 categories, rename column to "PavedStreet"
df["Street"] = df["Street"].map({"Pave": 1, "Grvl": 0})
df["Street"].rename("PavedStreet", inplace=True)
#Convert "Alley" to dummy variable since it has 1 big category and 2 small category, that we can merge, rename column to "HasAlleyAccess"
df["Alley"] = df["Alley"].map({"None": 0, "Grvl": 1, "Pave": 1})
df["Alley"].rename("HasAlleyAccess", inplace=True)
#%%
#Lets look at a distubition of SalePrice based on LotShape to see if we can merge IR1 with Reg and IR2 with IR3
plt.figure(figsize=(10, 6))
sns.boxplot(x="LotShape", y="SalePrice", data=df)
plt.title("SalePrice distribution by LotShape")
plt.xlabel("LotShape")
plt.ylabel("SalePrice")
# %% Lets create two categories for LotShape: Regular (1) and Irregular (0)
df["LotShape"] = df["LotShape"].map({"Reg": 1, "IR1": 0, "IR2": 0, "IR3": 0})
df["LotShape"].rename("RegularLotShape", inplace=True)
#For utilities, we can merge all categories except for "AllPub" into "Other"
df["Utilities"] = df["Utilities"].map({"AllPub": 1, "NoSewr": 0, "NoSeWa": 0, "ELO": 0})
df["Utilities"].rename("AllPublicUtilities", inplace=True)
#%%
# Lets look at a distubition of SalePrice based on LotConfig to see if we can merge FR2 with FR3
plt.figure(figsize=(10, 6))
sns.boxplot(x="LotConfig", y="SalePrice", data=df)
plt.title("SalePrice distribution by LotConfig")
plt.xlabel("LotConfig")
plt.ylabel("SalePrice") 

#The median SalePrice and distribution for FR2 and FR3 is similar, lets merge them into one category "MultipleFrontage"
df["LotConfig"] = df["LotConfig"].map({"Inside": "Inside", "Corner": "Corner", "CulDSac": "CulDSac", "FR2": "MultipleFrontage", "FR3": "MultipleFrontage"})
# One hot encode the categories for LotConfig with patsy, we can drop the first category to avoid multicollinearity
lotconfig_dummies = patsy.dmatrix("0 + C(LotConfig, treatment(reference='Inside'))", data=df, return_type="dataframe")
# For landslope we create a binary variable "GentleSlope" where Gentle = 1 and the rest = 0 since the distribution of SalePrice for Moderate and Severe is similar
df["GentleSlope"] = df["LandSlope"].map({"Gtl": 1, "Mod": 0, "Sev": 0})
# %% Lets look at negihbourhoods and see if we can group them based on SalePrice distribution
plt.figure(figsize=(12, 8))
sns.boxplot(x="Neighborhood", y="SalePrice", data=df)
plt.xticks(rotation=90)
plt.title("SalePrice distribution by Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("SalePrice")


# %%
s