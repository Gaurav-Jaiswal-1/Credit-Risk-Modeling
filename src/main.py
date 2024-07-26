# Importing libraries
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
import os

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor



# Load datasets
data1 = pd.read_excel(r"C:\Users\Gaurav\OneDrive\Desktop\Projects\Machine Learning Projects\Credit Risk Modeling (ML)\Datasets\case_study1.xlsx")
data2 = pd.read_excel(r"C:\Users\Gaurav\OneDrive\Desktop\Projects\Machine Learning Projects\Credit Risk Modeling (ML)\Datasets\case_study2.xlsx")


df1 = data1.copy()
df2 = data2.copy()


df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]


# Removing null values
columns_to_be_removed = []
for i in df2.columns:
    # shape[0] is for rows
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)

# axis = 1 shows is for columns
df2 = df2.drop(columns_to_be_removed,axis = 1)


for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]


# Merging two dataframes, df1 and df2, using the common column "PROSPECTID"
# The 'how' parameter specifies the type of merge to perform, which is 'inner' in this case.
# An 'inner' merge keeps only the rows with matching "PROSPECTID" in both dataframes.

df = pd.merge(df1, df2, how='inner', left_on=["PROSPECTID"], right_on=['PROSPECTID']) 

# - 'df': The resulting dataframe after merging.
# - 'df1' and 'df2': The two dataframes being merged.
# - 'how="inner"': This type of merge only includes rows with matching values in "PROSPECTID" from both dataframes.
# - 'left_on' and 'right_on': These specify the column to match on, which is "PROSPECTID" in both dataframes.



# Loop through each variable in the specified list
for i in ["MARITALSTATUS", "EDUCATION", "GENDER", "last_prod_enq2", "first_prod_enq2"]:
    # Calculate the Chi-Square statistic and p-value for the current variable 'i'
    # using a contingency table created from the dataframe 'df'
    # The contingency table compares the categories of the variable 'i' with 'Approved_Flag'

    # In Python, the underscore _ is often used as a placeholder for values that are not needed or used in the code
    # the chi2_contingency function returns four values:

    # chi2: The Chi-Square statistic.
    # pval: The p-value of the test.
    # _: The degrees of freedom (not used in this case, hence the underscore).
    # _: The expected frequencies table (also not used here, hence another underscore).
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df["Approved_Flag"]))
    
    # Print the variable name and the corresponding p-value from the Chi-Square test
    print(i, "----", pval)


numeric_columns = []
for i in df.columns:
    # Check if the data type of the column is not 'object' (i.e., it's a numeric type)
    # and also ensure that the column name is not in the list of excluded columns
    if df[i].dtype != "object" and i not in ["PROSPECTID","Approved_Flag"]:
        numeric_columns.append(i)


# Create a new DataFrame containing only the numeric columns identified earlier
vif_data = df[numeric_columns]

# Get the total number of numeric columns
total_columns = vif_data.shape[1]

# Initialize an empty list to keep track of the columns that will be kept
columns_to_be_kept = []

# Initialize the column index to start from 0
column_index = 0

# Loop through each index from 0 to the total number of numeric columns
for i in range(0, total_columns):
    # Calculate the Variance Inflation Factor (VIF) for the column at the current index
    # 'variance_inflation_factor' function requires the DataFrame and the index of the column to calculate VIF
    vif_value = variance_inflation_factor(vif_data, column_index)
    
    # Print the index of the column and its VIF value
    # print(column_index, "---", vif_value)

    # Check if the VIF value is within an acceptable range
    if vif_value <= 6:
        # If the VIF is acceptable, add the column name to the list of columns to be kept
        columns_to_be_kept.append(numeric_columns[i])
        # Increment the column index to move to the next column
        column_index = column_index + 1
    else:
        # If the VIF is too high, drop the column from the DataFrame
        # The column is removed from 'vif_data' using its name
        vif_data = vif_data.drop([numeric_columns[i]], axis=1)



# Check Anova for columns_to_be_kept
# Import the f_oneway function from scipy.stats for performing ANOVA
from scipy.stats import f_oneway

# Initialize an empty list to store the names of columns that will be kept after ANOVA testing
columns_to_be_kept_numerical = []

# Loop through each column name in the list 'columns_to_be_kept'
for i in columns_to_be_kept:
    # Convert the column values and 'Approved_Flag' values into lists
    a = list(df[i])
    b = list(df["Approved_Flag"])

    # Separate the data into groups based on 'Approved_Flag' values
    group_p1 = [value for value, group in zip(a, b) if group == 'P1']
    group_p2 = [value for value, group in zip(a, b) if group == 'P2']
    group_p3 = [value for value, group in zip(a, b) if group == 'P3']
    group_p4 = [value for value, group in zip(a, b) if group == 'P4']

    # Perform ANOVA test across the four groups
    f_statistics, p_value = f_oneway(group_p1, group_p2, group_p3, group_p4)

    # If the p-value is less than or equal to 0.05, it indicates a significant difference between groups
    # Therefore, the column is added to 'columns_to_be_kept_numerical'
    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)


# Assigning numerical values as per the education level
df.loc[df['EDUCATION'] == "SSC", ["EDUCATION"]]                 = 1
df.loc[df['EDUCATION'] == "12TH", ["EDUCATION"]]                = 2
df.loc[df['EDUCATION'] == "GRADUATE", ["EDUCATION"]]            = 3
df.loc[df['EDUCATION'] == "UNDER GRADUATE", ["EDUCATION"]]      = 3
df.loc[df['EDUCATION'] == "POST-GRADUATE", ["EDUCATION"]]       = 4
df.loc[df['EDUCATION'] == "OTHERS", ["EDUCATION"]]              = 1
df.loc[df['EDUCATION'] == "PROFESSIONAL", ["EDUCATION"]]        = 3



# his line displays the count of unique values in the EDUCATION column of the DataFrame df. 
# It helps to understand the distribution of different education levels in the dataset.
df['EDUCATION'].value_counts()

# This line converts the values in the EDUCATION column to integers
df['EDUCATION'] = df['EDUCATION'].astype(int)
# df.info()


# Import the pandas library
import pandas as pd

# Assume df is your original DataFrame with categorical columns

# Apply one-hot encoding to the specified categorical columns in df
# The function pd.get_dummies() converts categorical data into a set of binary columns
# Each unique value in the specified columns will become a separate column in the new DataFrame df_encoded
# The original categorical columns 'MARITALSTATUS', 'GENDER', 'last_prod_enq2', and 'first_prod_enq2' 
# will be replaced by the new binary columns
df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

# df_encoded is now a new DataFrame containing all the original columns from df except the specified 
# categorical columns, which have been replaced by their one-hot encoded binary columns
# Each original categorical value is represented by a separate column with binary values (0 or 1)


# Machine learning model fitting using Random forest
y = df_encoded["Approved_Flag"]
x = df_encoded.drop(["Approved_Flag"], axis=1)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42) 


# Initialize the Random Forest classifier
# n_estimators=200 sets the number of trees in the forest to 200
# random_state=42 ensures reproducibility by setting the seed for random number generation
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)


# Training
rf_classifier.fit(x_train, y_train)


y_pred = rf_classifier.predict(x_test)


# Checking accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print()
print(f"Accuracy: {accuracy}")
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)



for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class: {v}")
    print(f"Precision : {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()



import pickle

# Save the model 
filename = "customer_p_rank.sav"
pickle.dump(rf_classifier, open(filename,'wb'))




