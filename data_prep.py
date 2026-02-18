#%% Importing libraries
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")



def plot_categorical_summary(df):
    """
    Plots a grid of value counts and SalePrice distributions for all categorical variables.
    """
    cat_cols = df.select_dtypes(include="object").columns
    n = len(cat_cols)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4*n))
    for i, col in enumerate(cat_cols):
        sns.countplot(x=col, data=df, ax=axes[i,0])
        axes[i,0].set_title(f"{col} value counts")
        axes[i,0].tick_params(axis='x', rotation=45)
        sns.boxplot(x=col, y="SalePrice", data=df, ax=axes[i,1])
        axes[i,1].set_title(f"SalePrice by {col}")
        axes[i,1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("plots/categoricals_overview.png")

#%% 
df_train = pd.read_csv("data/train.csv")
df_train.shape
df_train.info()
plot_categorical_summary(df_train)
# %%
def clean_data(df):
    #The MSSubClass is a categorical variable but it is encoded as an integer, we can convert it to a string to treat it as a categorical variable
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df["LotFrontage"] = df.groupby("Neighborhood") ["LotFrontage"].transform(lambda x: x.fillna(x.median()))
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
    # Since our train data is only 1460 rows, we dont want to convert all the categorical variables into dummies, 
    # let's see value counts for each categorical variable and decide which ones to convert into dummies
    # Lets aim for a maximum of 150 features
    # --- IMPUTATION ---
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    df["Electrical"].fillna("SBrkr", inplace=True)
    df["MasVnrArea"].fillna(0, inplace=True)
    df["GarageYrBlt"].fillna(df["YearBuilt"], inplace=True)
    columns_to_impute_none = [
        "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"
    ]
    df[columns_to_impute_none] = df[columns_to_impute_none].fillna("None")

    # --- MAPPING & FEATURE ENGINEERING ---
    # Simple binary/categorical mappings
    df["Street"] = df["Street"].map({"Pave": 1, "Grvl": 0})
    df["Alley"] = df["Alley"].map({"None": 0, "Grvl": 1, "Pave": 1})
    df["LotShape"] = df["LotShape"].map({"Reg": 1, "IR1": 0, "IR2": 0, "IR3": 0})
    df["Utilities"] = df["Utilities"].map({"AllPub": 1, "NoSewr": 0, "NoSeWa": 0, "ELO": 0})
    df["GentleSlope"] = df["LandSlope"].map({"Gtl": 1, "Mod": 0, "Sev": 0})

    # Grouping/merging categories
    df["LotConfig"] = df["LotConfig"].map({"Inside": "Inside", "Corner": "Corner", "CulDSac": "CulDSac", "FR2": "MultipleFrontage", "FR3": "MultipleFrontage"})
    df["Neighborhood"] = df["Neighborhood"].replace(["Blueste", "NPkVill", "BrDale", "MeadowV"], "OtherLowValueNeighborhoods")
    df["Condition1"] = df["Condition1"].replace(["RRNn", "RRAn", "RRNe", "RRAe"], "NearRR")
    df["Condition2"] = df["Condition2"].replace(["RRNn", "RRAn", "RRNe", "RRAe"], "NearRR")
    df["BldgType"] = df["BldgType"].replace(["2fmCon", "Duplex"], "TwoFamilyDuplex")
    df["RoofStyle"] = df["RoofStyle"].replace(["Flat", "Gambrel", "Mansard", "Shed"], "OtherRoofStyle")
    df["Exterior1st"] = df["Exterior1st"].replace(["AsphShn", "BrkComm", "Stone", "ImStucc", "CBlock", "Other"], "OtherExterior1st")
    df["Exterior2nd"] = df["Exterior2nd"].replace(["AsphShn", "BrkComm", "Stone", "ImStucc", "CBlock", "Other"], "OtherExterior1st")
    df["Foundation"] = df["Foundation"].replace(["Slab", "Stone", "Wood"], "OtherFoundation")

    # Ordinal mappings
    df["ExterQual"] = df["ExterQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
    df["ExterCond"] = df["ExterCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
    df["BsmtQual"] = df["BsmtQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
    df["BsmtCond"] = df["BsmtCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
    df["BsmtExposure"] = df["BsmtExposure"].map({"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0})
    df["BsmtFinType1"] = df["BsmtFinType1"].map({"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0})
    df["BsmtFinType2"] = df["BsmtFinType2"].map({"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0})
    df["KitchenQual"] = df["KitchenQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
    df["FireplaceQu"] = df["FireplaceQu"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
    df["GarageFinish"] = df["GarageFinish"].map({"None": 0, "Unf": 1, "RFn": 2, "Fin": 3})
    df["GarageQual"] = df["GarageQual"].map({"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5})
    df["GarageCond"] = df["GarageCond"].map({"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5})
    df["PoolQC"] = df["PoolQC"].map({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "None": 0})

    # Binary/indicator features
    df["Heating_GasA"] = df["Heating"].map({"GasA": 1, "GasW": 0, "Grav": 0, "Wall": 0, "OthW": 0, "Floor": 0})
    df["GoodHeatingQC"] = df["HeatingQC"].map({"Ex": 1, "Gd": 0, "TA": 0, "Fa": 0, "Po": 0})
    df["HasCentralAir"] = df["CentralAir"].map({"Y": 1, "N": 0})
    df["Electrical_SBrkr"] = df["Electrical"].map({"SBrkr": 1, "FuseA": 0, "FuseF": 0, "FuseP": 0, "Mix": 0})
    df["Functional_Typ"] = df["Functional"].map({"Typ": 1, "Min1": 0, "Min2": 0, "Mod": 0, "Maj1": 0, "Maj2": 0, "Sev": 0, "Sal": 0})
    df["HasAttachedOrBuiltInGarage"] = df["GarageType"].map({"None": 0, "2Types": 0, "Detchd": 0, "CarPort": 0, "Basment": 0, "BuiltIn": 1, "Attchd": 1})
    df["HasPavedDriveway"] = df["PavedDrive"].map({"Y": 1, "N": 0, "P": 0})
    df["SaleType_New"] = df["SaleType"].map({"New": 1, "WD": 0, "CWD": 0, "VWD": 0, "COD": 0, "ConLD": 0, "ConLI": 0, "ConLw": 0, "Oth": 0})
    df["MSZoning_RL"] = df["MSZoning"].map({"RL": 1, "RM": 0, "C (all)": 0, "FV": 1, "RH": 0})

    # MSSubClass grouping
    df["MSSubClass"] = df["MSSubClass"].map({
        "20": "LowEnd", "30": "LowEnd", "40": "Average", "45": "LowEnd", "50": "Average", "60": "HighEnd", "70": "Average", 
        "75": "HighEnd", "80": "Average", "85": "Average", "90": "Average", "120": "HighEnd", "150": "Average", 
        "160": "Average", "180": "LowEnd", "190": "Average"
    })

    # --- DUMMY CREATION (before dropping columns) ---
    lotconfig_dummies = patsy.dmatrix("0 + C(LotConfig, Treatment(reference='Inside'))", data=df, return_type="dataframe")
    neighborhood_dummies = patsy.dmatrix("0 + C(Neighborhood, Treatment(reference='OtherLowValueNeighborhoods'))", data=df, return_type="dataframe")
    condition1_dummies = patsy.dmatrix("0 + C(Condition1, Treatment(reference='Norm'))", data=df, return_type="dataframe")
    condition2_dummies = patsy.dmatrix("0 + C(Condition2, Treatment(reference='Norm'))", data=df, return_type="dataframe")
    bldgtype_dummies = patsy.dmatrix("0 + C(BldgType, Treatment(reference='1Fam'))", data=df, return_type="dataframe")
    housestyle_dummies = patsy.dmatrix("0 + C(HouseStyle, Treatment(reference='1Story'))", data=df, return_type="dataframe")
    roofstyle_dummies = patsy.dmatrix("0 + C(RoofStyle, Treatment(reference='Gable'))", data=df, return_type="dataframe")
    exterior1st_dummies = patsy.dmatrix("0 + C(Exterior1st, Treatment(reference='VinylSd'))", data=df, return_type="dataframe")
    exterior2nd_dummies = patsy.dmatrix("0 + C(Exterior2nd, Treatment(reference='VinylSd'))", data=df, return_type="dataframe")
    masvnrtype_dummies = patsy.dmatrix("0 + C(MasVnrType, Treatment(reference='None'))", data=df, return_type="dataframe")
    foundation_dummies = patsy.dmatrix("0 + C(Foundation, Treatment(reference='PConc'))", data=df, return_type="dataframe")
    class_dummies = patsy.dmatrix("0 + C(MSSubClass, Treatment(reference='Average'))", data=df, return_type="dataframe")

    # --- CONCAT DUMMIES ---
    df = pd.concat([
        df, lotconfig_dummies, condition1_dummies, condition2_dummies, bldgtype_dummies, housestyle_dummies, 
        roofstyle_dummies, exterior1st_dummies, exterior2nd_dummies, masvnrtype_dummies, foundation_dummies, 
        neighborhood_dummies, class_dummies
    ], axis=1)

    # --- DROP COLUMNS (batch) ---
    drop_cols = [
        "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", 
        "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "HeatingQC", "CentralAir", 
        "Electrical", "Functional", "GarageType", "PavedDrive", "Fence", "MiscFeature", "SaleType", "SaleCondition", 
        "MSZoning", "MSSubClass", "LandContour"
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    print(df.info())
    print(df.shape)
    return df

#%%
# Clean test and train data
df_train_clean = clean_data(df_train)

df_test = pd.read_csv("data/test.csv")
df_test_clean = clean_data(df_test)

# %%
df_train_clean.columns.difference(df_test_clean.columns) #I noticed that column numbers are different, so lets find out why
# %%
df_train_clean.drop(columns=["C(Condition2, Treatment(reference='Norm'))[NearRR]", 
                             "C(HouseStyle, Treatment(reference='1Story'))[2.5Fin]"], inplace=True)

print("Train set size: ", df_train_clean.shape)
print("Test set size: ", df_test_clean.shape)
#%%
df_test_clean.to_csv("data/transformed_test.csv", index=False)
df_train_clean.to_csv("data/transformed_train.csv", index=False)