Week 1 Task: Data Acquisition, Cleaning, and Preprocessing
Overview

This project demonstrates the process of acquiring a publicly available dataset, exploring its structure and quality, and cleaning/preprocessing it using Python. The goal was to prepare a real-world dataset for further analysis by systematically handling missing values, outliers, and erroneous entries.

Dataset
Name: Titanic Passenger Dataset
Source: Public GitHub repository (datasciencedojo/datasets)
Description: Contains demographic and travel details of Titanic passengers, including survival outcome.
What Was Done
Data Acquisition — Loaded the raw CSV dataset into a pandas DataFrame.
Initial Exploration — Checked shape, data types, summary statistics, and missing values.
Missing Value Handling
Age → filled with median
Embarked → filled with mode
Cabin → dropped (80% missing), replaced with a binary HasCabin feature
Outlier Detection & Treatment — Detected outliers in Fare using the IQR method and treated them via capping (winsorization).
Erroneous Entry Checks — Verified no negative values or duplicate rows existed.
Feature Engineering — Extracted Title from names, created FamilySize, and encoded categorical variables (Sex, Embarked).

Files in This Repository
File
analysis.py
Python script containing the full data cleaning and preprocessing pipeline
titanic_cleaned.csv
Final cleaned dataset, ready for further analysis
Week1_Data_Cleaning_Report.docx
Full written report with explanations, code snippets, and visualizations

Tools & Libraries
pandas, numpy, matplotlib, seaborn

How to Run
pip install pandas numpy matplotlib seaborn
python analysis.py

This will regenerate titanic_cleaned.csv and the supporting plots.
