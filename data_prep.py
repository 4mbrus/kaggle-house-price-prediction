#%% Importing libraries
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


def quick_analytics(categorical_var):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=f"{categorical_var}", y="SalePrice", data=df)
    print(df[f"{categorical_var}"].value_counts())

#%% Data exploration
df = pd.read_csv("data/train.csv")
df.shape
df.info()
# %%
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
quick_analytics("MSSubClass")



#%%
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
lotconfig_dummies = patsy.dmatrix("0 + C(LotConfig, Treatment(reference='Inside'))", data=df, return_type="dataframe")
df.drop("LotConfig", axis=1, inplace=True)
# For landslope we create a binary variable "GentleSlope" where Gentle = 1 and the rest = 0 since the distribution of SalePrice for Moderate and Severe is similar
df["GentleSlope"] = df["LandSlope"].map({"Gtl": 1, "Mod": 0, "Sev": 0})
df.drop("LandSlope", axis=1, inplace=True)
# %% Lets look at negihbourhoods and see if we can group them based on SalePrice distribution
plt.figure(figsize=(12, 8))
sns.boxplot(x="Neighborhood", y="SalePrice", data=df)
plt.xticks(rotation=90)
plt.title("SalePrice distribution by Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("SalePrice")
df["Neighborhood"].value_counts()
# %% Merge some of the low price, low value count neighbourhoods into "OtherLowValueNeighborhoods"
# To merge: Blueste, NPkVill, BrDale, MeadowV
df["Neighborhood"] = df["Neighborhood"].replace(["Blueste", "NPkVill", "BrDale", "MeadowV"], "OtherLowValueNeighborhoods")
neighborhood_dummies = patsy.dmatrix("0 + C(Neighborhood, Treatment(reference='OtherLowValueNeighborhoods'))", data=df, return_type="dataframe")
df.drop("Neighborhood", axis=1, inplace=True)
#For condition1 we merge railroad categories to "NearRR"
df["Condition1"] = df["Condition1"].replace(["RRNn", "RRAn", "RRNe", "RRAe"], "NearRR")
df["Condition2"] = df["Condition2"].replace(["RRNn", "RRAn", "RRNe", "RRAe"], "NearRR")
condition1_dummies = patsy.dmatrix("0 + C(Condition1, Treatment(reference='Norm'))", data=df, return_type="dataframe")
condition2_dummies = patsy.dmatrix("0 + C(Condition2, Treatment(reference='Norm'))", data=df, return_type="dataframe")
df.drop(["Condition1", "Condition2"], axis=1, inplace=True)
#%% BldgType seems like an important variable, lets look at the distribution of SalePrice based on BldgType to see if we can merge some categories
quick_analytics("BldgType") #2 family conversion and duplex have similar distribution and median SalePrice, we can merge them into "TwoFamilyDuplex"
df["BldgType"] = df["BldgType"].replace(["2fmCon", "Duplex"], "TwoFamilyDuplex")
bldgtype_dummies = patsy.dmatrix("0 + C(BldgType, Treatment(reference='1Fam'))", data=df, return_type="dataframe")
df.drop("BldgType", axis=1, inplace=True)
# %%
quick_analytics("HouseStyle") #no clear mergeable categories, we can one hot encode all categories with patsy, we can drop the first category to avoid multicollinearity
housestyle_dummies = patsy.dmatrix("0 + C(HouseStyle, Treatment(reference='1Story'))", data=df, return_type="dataframe")
df.drop("HouseStyle", axis=1, inplace=True)
# %%
quick_analytics("RoofStyle") # Merge small categories into "OtherRoofStyle"
df["RoofStyle"] = df["RoofStyle"].replace(["Flat", "Gambrel", "Mansard", "Shed"], "OtherRoofStyle")
roofstyle_dummies = patsy.dmatrix("0 + C(RoofStyle, Treatment(reference='Gable'))", data=df, return_type="dataframe")
df.drop("RoofStyle", axis=1, inplace=True)
# %%
quick_analytics("RoofMatl") #categories are too small, exclude this variable from the model
df.drop("RoofMatl", axis=1, inplace=True)
# %%
quick_analytics("Exterior1st") #merge small categories into "OtherExterior1st"
df["Exterior1st"] = df["Exterior1st"].replace(["AsphShn", "BrkComm", "Stone", "ImStucc", "CBlock", "Other"], "OtherExterior1st")
exterior1st_dummies = patsy.dmatrix("0 + C(Exterior1st, Treatment(reference='VinylSd'))", data=df, return_type="dataframe")
df.drop("Exterior1st", axis=1, inplace=True)
# %%
quick_analytics("Exterior2nd")
df["Exterior2nd"] = df["Exterior2nd"].replace(["AsphShn", "BrkComm", "Stone", "ImStucc", "CBlock", "Other"], "OtherExterior1st")
exterior2nd_dummies = patsy.dmatrix("0 + C(Exterior2nd, Treatment(reference='VinylSd'))", data=df, return_type="dataframe")
df.drop("Exterior2nd", axis=1, inplace=True)
# %%
quick_analytics("MasVnrType") #encode with dummies, we can drop the first category to avoid multicollinearity
masvnrtype_dummies = patsy.dmatrix("0 + C(MasVnrType, Treatment(reference='None'))", data=df, return_type="dataframe")
df.drop("MasVnrType", axis=1, inplace=True)
# %%
quick_analytics("ExterQual") #make into continous variable 5-Excellent, 4-Good, 3-Typical, 2-Fair, 1-Poor
df["ExterQual"] = df["ExterQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
# do the same for ExterCond
df["ExterCond"] = df["ExterCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
# %%
quick_analytics("Foundation") 
# %% Merge small categories into "OtherFoundation"
df["Foundation"] = df["Foundation"].replace(["Slab", "Stone", "Wood"], "OtherFoundation")
foundation_dummies = patsy.dmatrix("0 + C(Foundation, Treatment(reference='PConc'))", data=df, return_type="dataframe")
df.drop("Foundation", axis=1, inplace=True)
#%% For BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2 we can turn them into continous variables
# BsmtQual and BsmtCond: 5-Excellent, 4-Good, 3-Typical, 2-Fair, 1-Poor, 0-No basement
df["BsmtQual"] = df["BsmtQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
df["BsmtCond"] = df["BsmtCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
# BsmtExposure: 4-Good exposure, 3-Average exposure, 2-Little exposure, 1-No exposure, 0-No basement
df["BsmtExposure"] = df["BsmtExposure"].map({"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0})
# BsmtFinType1 and BsmtFinType2: 6-GLQ, 5-ALQ, 4-BLQ, 3-Rec, 2-LwQ, 1-Unf, 0-No basement
df["BsmtFinType1"] = df["BsmtFinType1"].map({"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0})
df["BsmtFinType2"] = df["BsmtFinType2"].map({"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0})
#%%
quick_analytics("Heating") #Create binary variable "Heating_GasA"
df["Heating_GasA"] = df["Heating"].map({"GasA": 1, "GasW": 0, "Grav": 0, "Wall": 0, "OthW": 0, "Floor": 0})
df.drop("Heating", axis=1, inplace=True)
# %%
quick_analytics("HeatingQC") # Crate binary variable "GoodHeatingQC" where Ex = 1 and the rest = 0 since the distribution of SalePrice for TA, Fa and Po is similar
df["GoodHeatingQC"] = df["HeatingQC"].map({"Ex": 1, "Gd": 0, "TA": 0, "Fa": 0, "Po": 0})
df.drop("HeatingQC", axis=1, inplace=True)
# %%
quick_analytics("CentralAir") # Binary variable, map Y to 1 and N to 0, rename column to "HasCentralAir"
df["HasCentralAir"] = df["CentralAir"].map({"Y": 1, "N": 0})
df.drop("CentralAir", axis=1, inplace=True)
# %%
quick_analytics("Electrical") # Unbalanced variable, we can create a binary variable "Electrical_SBrkr" where SBrkr = 1 and the rest = 0
df["Electrical_SBrkr"] = df["Electrical"].map({"SBrkr": 1, "FuseA": 0, "FuseF": 0, "FuseP": 0, "Mix": 0})
df.drop("Electrical", axis=1, inplace=True)
# %%
quick_analytics("KitchenQual") # Create continuous variable where Ex = 5, Gd = 4, TA = 3, Fa = 2, Po = 1
df["KitchenQual"] = df["KitchenQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
# %%
quick_analytics("Functional") # Create binary variable "Functional_Typ" where Typ = 1 and the rest = 0, categories are very unbalanced and the distribution of SalePrice for the rest of the categories is more or less similar
df["Functional_Typ"] = df["Functional"].map({"Typ": 1, "Min1": 0, "Min2": 0, "Mod": 0, "Maj1": 0, "Maj2": 0, "Sev": 0, "Sal": 0})
df.drop("Functional", axis=1, inplace=True)
# %%
quick_analytics("FireplaceQu") # Create continuous variable where Ex = 5, Gd = 4, TA = 3, Fa = 2, Po = 1, None = 0
df["FireplaceQu"] = df["FireplaceQu"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
# %%
quick_analytics("GarageType") # Create binary Attchd and BuiltIn categories since they have higher median SalePrice than the rest of the categories, rename column to "HasAttachedOrBuiltInGarage"
df["HasAttachedOrBuiltInGarage"] = df["GarageType"].map({"None": 0, "2Types": 0, "Detchd": 0, "CarPort": 0, "Basment": 0, "BuiltIn": 1, "Attchd": 1})
df.drop("GarageType", axis=1, inplace=True)
# %%
quick_analytics("GarageFinish") # Convert to numerical
df["GarageFinish"] = df["GarageFinish"].map({"None": 0, "Unf": 1, "RFn": 2, "Fin": 3})
# %%
quick_analytics("GarageQual") # Convert to numerical
df["GarageQual"] = df["GarageQual"].map({"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5})
# %%
quick_analytics("GarageCond") # Convert to numerical
df["GarageCond"] = df["GarageCond"].map({"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5})
# %%
quick_analytics("PavedDrive") # Create binary variable where Y = 1 and N/P = 0, rename column to "HasPavedDriveway"
df["HasPavedDriveway"] = df["PavedDrive"].map({"Y": 1, "N": 0, "P": 0})
df.drop("PavedDrive", axis=1, inplace=True)
# %%
quick_analytics("PoolQC") # Convert to numerical variable where Ex = 5, Gd = 4, TA = 3, Fa = 2, Po = 1, None = 0
df["PoolQC"] = df["PoolQC"].map({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "None": 0})
# %%
quick_analytics("Fence") # Unvbalanced, no clear hierarchy, better to drop this variable
df.drop("Fence", axis=1, inplace=True)
# %%
quick_analytics("MiscFeature") #Unbalanced, no clear hierarchy, better to drop this variable
df.drop("MiscFeature", axis=1, inplace=True)
# %%
quick_analytics("SaleType") # New is the clear outlier, we can create a binary variable "SaleType_New" where New = 1 and the rest = 0, rename column to "SaleType_New"
df["SaleType_New"] = df["SaleType"].map({"New": 1, "WD": 0, "CWD": 0, "VWD": 0, "COD": 0, "ConLD": 0, "ConLI": 0, "ConLw": 0, "Oth": 0})
df.drop("SaleType", axis=1, inplace=True)
# %%
quick_analytics("SaleCondition") # Partial probably captures the same pattern as SaleType_New, other categories are unbalanced and have similar distribution, we can drop this variable
df.drop("SaleCondition", axis=1, inplace=True)
#%%
quick_analytics("MSZoning") # Create binary variable "MSZoning_RL" where RL, FV= 1 and the rest = 0 since RL and FV have higher median SalePrice than the rest of the categories, rename column to "MSZoning_RL"
df["MSZoning_RL"] = df["MSZoning"].map({"RL": 1, "RM": 0, "C (all)": 0, "FV": 1, "RH": 0})
df.drop("MSZoning", axis=1, inplace=True)
# %% Sanity cheks for the transformed df
print(df.info())
print(df.shape)

# %% merge dummies together and then merge with the original df
df = pd.concat([df, lotconfig_dummies, condition1_dummies, condition2_dummies, bldgtype_dummies, housestyle_dummies, roofstyle_dummies, exterior1st_dummies, exterior2nd_dummies, masvnrtype_dummies, foundation_dummies, neighborhood_dummies], axis=1)

# %% Sanity cheks for the transformed df
print(df.info())
print(df.shape)