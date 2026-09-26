import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------- STEP 1: DATA ACQUISITION ----------
df = pd.read_csv("titanic.csv")
print("STEP 1: DATA ACQUISITION")
print("Shape:", df.shape)
print(df.head())

# ---------- STEP 2: INITIAL EXPLORATION ----------
print("\nSTEP 2: INITIAL EXPLORATION")
print(df.info())
print("\nSummary statistics:\n", df.describe(include="all"))

missing_before = df.isnull().sum()
missing_pct = (missing_before / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing_before, "missing_pct": missing_pct})
missing_report = missing_report[missing_report.missing_count > 0]
print("\nMissing values BEFORE cleaning:\n", missing_report)

# Plot 1: Missing values bar chart
plt.figure(figsize=(6, 4))
missing_report["missing_pct"].plot(kind="bar", color="#d9534f")
plt.title("Missing Values (%) Before Cleaning")
plt.ylabel("% missing")
plt.tight_layout()
plt.savefig("plot_missing_before.png", dpi=140)
plt.close()

# ---------- STEP 3: HANDLING MISSING VALUES ----------
print("\nSTEP 3: HANDLING MISSING VALUES")

# Age -> fill with median (robust to outliers, keeps distribution shape reasonable)
median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)
print(f"Filled {missing_before['Age']} missing Age values with median = {median_age}")

# Embarked -> fill with mode (most frequent port)
mode_embarked = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(mode_embarked)
print(f"Filled {missing_before.get('Embarked',0)} missing Embarked values with mode = '{mode_embarked}'")

# Cabin -> extremely high missing %, so instead of imputing we engineer a binary flag
df["HasCabin"] = df["Cabin"].notnull().astype(int)
df.drop(columns=["Cabin"], inplace=True)
print(f"Dropped 'Cabin' column ({missing_before['Cabin']} missing) and created binary feature 'HasCabin' instead")

missing_after = df.isnull().sum().sum()
print(f"\nTotal missing values AFTER cleaning: {missing_after}")

# ---------- STEP 4: OUTLIER DETECTION & TREATMENT (Fare) ----------
print("\nSTEP 4: OUTLIER DETECTION (Fare) using IQR method")
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["Fare"] < lower_bound) | (df["Fare"] > upper_bound)]
print(f"Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
print(f"Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Number of outliers found in 'Fare': {len(outliers)}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(x=df["Fare"], color="#f0ad4e")
plt.title("Fare - Before Capping (Outliers Visible)")
# Cap outliers instead of deleting rows (preserves sample size / other info)
df["Fare"] = np.where(df["Fare"] > upper_bound, upper_bound, df["Fare"])
df["Fare"] = np.where(df["Fare"] < lower_bound, lower_bound, df["Fare"])
plt.subplot(1, 2, 2)
sns.boxplot(x=df["Fare"], color="#5cb85c")
plt.title("Fare - After Capping (Outliers Treated)")
plt.tight_layout()
plt.savefig("plot_outliers_fare.png", dpi=140)
plt.close()

# ---------- STEP 5: CORRECTING ERRONEOUS / INCONSISTENT ENTRIES ----------
print("\nSTEP 5: CHECKING FOR ERRONEOUS ENTRIES")
neg_age = (df["Age"] < 0).sum()
neg_fare = (df["Fare"] < 0).sum()
dupes = df.duplicated().sum()
print(f"Negative Age values: {neg_age}")
print(f"Negative Fare values: {neg_fare}")
print(f"Duplicate rows: {dupes}")
df.drop_duplicates(inplace=True)

# ---------- STEP 6: FEATURE ENGINEERING / PREPROCESSING ----------
print("\nSTEP 6: FEATURE ENGINEERING")

# Extract Title from Name (helps capture social status / age group cues)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
print("Extracted titles:", df["Title"].unique())

# Family size feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Encode categorical variables
df["Sex_encoded"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked")

print("\nFinal columns:", list(df.columns))
print("\nFinal shape:", df.shape)

plt.figure(figsize=(5, 4))
df["FamilySize"].value_counts().sort_index().plot(kind="bar", color="#5bc0de")
plt.title("Family Size Distribution (Engineered Feature)")
plt.xlabel("Family size")
plt.ylabel("Passenger count")
plt.tight_layout()
plt.savefig("plot_familysize.png", dpi=140)
plt.close()

# ---------- STEP 7: SAVE CLEANED DATASET ----------
df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as titanic_cleaned.csv")
