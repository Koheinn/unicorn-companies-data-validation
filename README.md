# Unicorn Companies — Data Validation & Categorical Encoding

A data analysis project completed as part of **Course 2 of the Google Advanced Data Analytics Professional Certificate**.

This project focuses on preparing a unicorn-company dataset for analysis by applying **data validation, data cleaning, feature engineering, categorical encoding, and basic exploratory analysis techniques using Python**.

The analysis is based on a dataset containing information about private companies that have reached unicorn status, including their valuation, industry, location, founding year, funding, and selected investors.

---

## 📌 Project Overview

In this activity, the goal is to help an investment firm better understand companies that have reached **unicorn status** — privately held companies valued at at least **$1 billion**.

The analysis pays particular attention to investment patterns and company characteristics that could help identify potential future investment opportunities.

The dataset contains information such as:

* Company name
* Valuation
* Date the company became a unicorn
* Industry
* City
* Country/Region
* Continent
* Year founded
* Funding
* Selected investors

The notebook demonstrates how raw data can be validated, cleaned, transformed, and converted into formats that are more suitable for statistical analysis and machine learning.

---

## 🎯 Objectives

The main objectives of this project are to:

1. Inspect the structure and data types of the dataset.
2. Convert columns to appropriate data types.
3. Create new analytical features.
4. Identify and correct invalid data.
5. Detect and remove duplicate companies.
6. Standardize inconsistent categorical labels.
7. Convert numerical data into categorical groups.
8. Apply different categorical encoding techniques.
9. Understand the advantages and limitations of different encoding methods.
10. Prepare the dataset for potential downstream analysis and machine learning.

---

## 🛠️ Technologies & Libraries

The project was implemented in Python using the following tools and libraries:

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Plotly**

### Main techniques used

* Data type inspection
* Data type conversion
* Datetime manipulation
* Feature engineering
* Descriptive statistics
* Input validation
* Data cleaning
* Duplicate detection
* Duplicate removal
* Categorical data normalization
* Quantile-based categorization
* Ordinal label encoding
* Nominal label encoding
* Dummy / one-hot encoding
* Grouping and aggregation

The notebook imports NumPy, pandas, seaborn, Matplotlib, Plotly, and `datetime` for the analysis.

---

## 📂 Repository Structure

```text
unicorn-companies-data-validation/
│
├── README.md
├── Activity_Validate_and_Clean_Your_Data.ipynb
│
└── Modified_Unicorn_Companies.csv
```

> **Note:** The notebook expects the CSV dataset to be located in the same directory as the notebook.

The notebook loads the dataset using:

```python
companies = pd.read_csv("Modified_Unicorn_Companies.csv")
```

---

## 📊 Dataset

The dataset contains information about unicorn companies and includes fields such as:

| Column             | Description                                    |
| ------------------ | ---------------------------------------------- |
| `Company`          | Name of the company                            |
| `Valuation`        | Company valuation in billions of USD           |
| `Date Joined`      | Date the company reached unicorn status        |
| `Industry`         | Industry/category of the company               |
| `City`             | City associated with the company               |
| `Country/Region`   | Country or region of the company               |
| `Continent`        | Continent where the company is located         |
| `Year Founded`     | Year the company was founded                   |
| `Funding`          | Total funding represented in the dataset       |
| `Select Investors` | Selected investors associated with the company |

The notebook's initial dataset inspection shows these ten original variables.

### Dataset Reference

The notebook identifies the following source:

**Bhat, M.A. — Unicorn Companies**

[Kaggle: Unicorn Companies](https://www.kaggle.com/datasets/mysarahmadbhat/unicorn-companies)

---

# 🔍 Analysis Workflow

## 1. Import Libraries

The project begins by importing the Python libraries required for data manipulation and visualization.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime as dt
```

---

## 2. Load and Inspect the Dataset

The dataset is loaded into a pandas DataFrame called `companies`.

```python
companies = pd.read_csv("Modified_Unicorn_Companies.csv")

companies.head()
```

The notebook also configures pandas to display all columns:

```python
pd.set_option("display.max_columns", None)
```

This makes it easier to inspect the complete structure of the dataset.

---

# 🧹 Data Cleaning

## 3. Inspect Data Types

The data types of all columns are examined using:

```python
companies.dtypes
```

The initial inspection shows that `Date Joined` is stored as an `object`, while numerical variables such as `Valuation` and `Year Founded` are stored as integers.

---

## 4. Convert `Date Joined` to Datetime

The `Date Joined` column is converted from a string/object into a pandas datetime type:

```python
companies["Date Joined"] = pd.to_datetime(
    companies["Date Joined"]
)
```

This allows the date information to be used more effectively for feature engineering and temporal analysis.

After conversion, the column has a `datetime64[ns]` data type.

---

# 🧮 Feature Engineering

## 5. Create `Year Joined`

The year in which each company achieved unicorn status is extracted from `Date Joined`.

```python
companies["Year Joined"] = companies["Date Joined"].dt.year
```

---

## 6. Create `Years To Unicorn`

A new feature called `Years To Unicorn` is created to represent the number of years between a company's founding and the year it became a unicorn.

```python
companies["Years To Unicorn"] = (
    companies["Year Joined"] - companies["Year Founded"]
)
```

This feature can help identify how quickly companies reached unicorn status.

Understanding how quickly a company achieves unicorn status can reveal trends and common characteristics that may be useful when evaluating potential future investments.

---

# ✅ Input Validation

The notebook identifies three major data-quality problems:

1. Incorrect data
2. Inconsistent industry labels
3. Duplicate companies

These issues are explicitly identified as problems that need to be corrected before continuing with the analysis.

---

## 7. Identify Invalid `Years To Unicorn` Values

Descriptive statistics are used to inspect the `Years To Unicorn` feature:

```python
companies["Years To Unicorn"].describe()
```

The original data contains a minimum value of `-3`.

A negative number of years is logically invalid because a company cannot achieve unicorn status before it was founded.

### Problem Identified

The invalid record belongs to:

**InVision**

The dataset originally indicated:

```text
Year Founded: 2020
Year Joined: 2017
Years To Unicorn: -3
```

The notebook corrects the founding year to **2011** based on external verification.

The `Years To Unicorn` feature is then recalculated:

```python
companies["Years To Unicorn"] = (
    companies["Year Joined"] - companies["Year Founded"]
)
```

Finally, the dataset is checked again to confirm that there are no remaining negative values.

---

# 🏷️ Standardizing Industry Labels

## 8. Validate Industry Categories

The notebook defines an approved list of industry labels:

```python
industry_list = [
    "Artificial intelligence",
    "Other",
    "E-commerce & direct-to-consumer",
    "Fintech",
    "Internet software & services",
    "Supply chain, logistics, & delivery",
    "Consumer & retail",
    "Data management & analytics",
    "Edtech",
    "Health",
    "Hardware",
    "Auto & transportation",
    "Travel",
    "Cybersecurity",
    "Mobile & telecommunications"
]
```

The dataset is then compared against this list to identify unexpected values.

Three inconsistent labels are identified:

```text
Artificial Intelligence
Data management and analytics
FinTech
```

These are treated as inconsistent versions of approved industry labels.

---

## 9. Correct Industry Labels

A replacement dictionary is created:

```python
replacement_dict = {
    "Artificial Intelligence": "Artificial intelligence",
    "Data management and analytics": "Data management & analytics",
    "FinTech": "Fintech"
}
```

The replacements are applied using pandas:

```python
companies["Industry"] = companies["Industry"].replace(
    replacement_dict
)
```

The dataset is then checked again to verify that all industry values belong to the approved `industry_list`.

---

# ♻️ Duplicate Data

## 10. Identify Duplicate Companies

The business requirement specifies that each company should appear only once.

Duplicate company names are identified using:

```python
companies[
    companies.duplicated(
        subset="Company",
        keep=False
    )
]
```

The notebook identifies duplicate entries for companies including:

* BrewDog
* ZocDoc
* SoundHound

## Some duplicate records contain small variations in fields such as location, industry, or selected investors.

## 11. Remove Duplicate Companies

Because the duplicated records represent the same companies rather than separate legitimate companies, only the first occurrence is retained.

```python
companies = companies.drop_duplicates(
    subset="Company",
    keep="first"
)
```

This ensures that every company appears only once in the cleaned dataset.

---

# 📈 Converting Numerical Data to Categorical Data

## 12. Create `High Valuation`

The `Valuation` column represents company valuation in billions of USD.

A new categorical feature called `High Valuation` is created:

* `low` → bottom 50% of valuations
* `high` → top 50% of valuations

This is implemented using pandas `qcut()`:

```python
companies["High Valuation"] = pd.qcut(
    companies["Valuation"],
    2,
    labels=["low", "high"]
)
```

## Using quantiles divides the observations into two equal-sized groups based on their valuation.

# 🔢 Categorical Encoding

A major learning objective of this activity is understanding how categorical variables can be transformed into numerical representations.

The notebook covers three approaches:

1. Ordinal label encoding
2. Nominal label encoding
3. Dummy / one-hot encoding

The appropriate approach depends on whether categories have an inherent hierarchy or should be treated as equally important.

---

## 13. Encode `Continent`

For this exercise, continents are treated as an **ordinal variable**.

The reasoning is based on an investment assumption: continents with fewer unicorn companies are given greater importance because they could potentially represent unrealized market opportunities.

The number of unicorn companies per continent is calculated.

### Result

| Rank | Continent     | Unicorn Companies |
| ---: | ------------- | ----------------: |
|    1 | North America |               586 |
|    2 | Asia          |               310 |
|    3 | Europe        |               143 |
|    4 | South America |                21 |
|    5 | Oceania       |                 8 |
|    6 | Africa        |                 3 |

These counts are produced directly in the notebook.

The resulting ordinal encoding is:

| Continent     | Numeric Value |
| ------------- | ------------: |
| North America |             1 |
| Asia          |             2 |
| Europe        |             3 |
| South America |             4 |
| Oceania       |             5 |
| Africa        |             6 |

A new column called `Continent Number` is created using a mapping dictionary.

---

## 14. Encode `Country/Region`

`Country/Region` is treated as a **nominal categorical variable** because there is no meaningful hierarchical order between countries.

Instead of creating a large number of dummy variables, label encoding is used:

```python
companies["Country/Region Numeric"] = (
    companies["Country/Region"]
    .astype("category")
    .cat.codes
)
```

Each unique country/region receives a numerical code.

---

## 15. Encode `Industry`

`Industry` is converted using **dummy encoding / one-hot encoding**.

```python
industry_data = pd.get_dummies(
    companies["Industry"]
)

companies = pd.concat(
    [companies, industry_data],
    axis=1
)
```

This creates a separate binary column for each industry category.

For example:

```text
Artificial intelligence
Auto & transportation
Consumer & retail
Cybersecurity
Fintech
Health
Hardware
...
```

Each company receives a `1` for its corresponding industry and `0` for the other industry categories.

---

# 🧠 Encoding Strategy

The encoding methods used in this project can be summarized as follows:

| Variable         | Data Type           | Encoding                 | Reason                                                                            |
| ---------------- | ------------------- | ------------------------ | --------------------------------------------------------------------------------- |
| `Continent`      | Ordinal categorical | Ordinal label encoding   | Categories are assigned a hierarchy based on the exercise's investment assumption |
| `Country/Region` | Nominal categorical | Label encoding           | Countries have no inherent hierarchical order                                     |
| `Industry`       | Nominal categorical | Dummy / one-hot encoding | Industries are treated as equally important categories                            |

The notebook explicitly explains these choices and their reasoning.

---

# ⚖️ Advantages & Disadvantages of Label Encoding

## Advantages

Label encoding converts categorical values into numerical values, making the resulting data compatible with machine learning algorithms that require numerical inputs.

For example:

```text
Country A → 1
Country B → 2
Country C → 3
```

This can be more compact than creating a separate binary column for every category.

---

## Disadvantages

Label encoding can make categorical values harder to interpret.

More importantly, assigning numerical values can unintentionally imply relationships or ordering between categories where none actually exists.

For example:

```text
Country A → 1
Country B → 2
Country C → 3
```

does **not** mean that Country C is "greater than" Country B or Country A.

The notebook highlights this potential issue as one of the disadvantages of label encoding.

---

# 🔎 Key Data Quality Issues Found

| Issue                        | Example                              | Resolution                                      |
| ---------------------------- | ------------------------------------ | ----------------------------------------------- |
| Invalid founding year        | InVision                             | Corrected `Year Founded`                        |
| Negative `Years To Unicorn`  | `-3`                                 | Recalculated after correcting the founding year |
| Inconsistent industry labels | `FinTech`, `Artificial Intelligence` | Standardized using replacement mapping          |
| Duplicate companies          | BrewDog, ZocDoc, SoundHound          | Retained first occurrence                       |
| Categorical variables        | Continent, Country/Region, Industry  | Converted using appropriate encoding techniques |

---

# 📚 Skills Demonstrated

This project demonstrates practical skills in:

### Python

* Python data structures
* Pandas DataFrames
* DataFrame filtering
* Series operations
* Dictionary-based mappings
* Datetime operations

### Data Cleaning

* Detecting invalid values
* Correcting incorrect records
* Standardizing categorical values
* Removing duplicate observations
* Validating data against predefined categories

### Feature Engineering

* Extracting year from dates
* Calculating `Years To Unicorn`
* Creating categorical valuation groups

### Categorical Data

* Ordinal encoding
* Nominal label encoding
* Dummy encoding
* One-hot encoding
* Understanding categorical variable types

### Exploratory Data Analysis

* `head()`
* `dtypes`
* `describe()`
* `groupby()`
* `agg()`
* `sort_values()`
* `value_counts()`

---

# 💡 Key Takeaways

Several important data-analysis concepts were reinforced through this activity:

* **Input validation is essential** for maintaining data quality.
* Incorrect data can produce misleading analytical results.
* Data cleaning often requires identifying logical inconsistencies rather than simply checking for missing values.
* Duplicate records can distort analysis and should be handled carefully.
* Categorical values need to be represented appropriately before they can be used by many machine learning algorithms.
* **Label encoding and one-hot/dummy encoding have different use cases.**
* The choice of encoding technique should depend on the nature of the categorical variable.
* Feature engineering can transform raw information into variables that are more useful for analysis.
* Data preparation is an important foundation for later statistical and machine learning work.

The notebook's final conclusion similarly emphasizes input validation, data quality, and choosing encoding methods on a case-by-case basis.

---

# 🚀 How to Run the Project

## 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

## 2. Navigate into the project

```bash
cd YOUR-REPOSITORY
```

## 3. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn plotly jupyter
```

## 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Activity_Validate_and_Clean_Your_Data.ipynb
```

Make sure the CSV file is in the same directory:

```text
Modified_Unicorn_Companies.csv
```

---

# 📁 Recommended Repository Structure

For GitHub, I recommend the following structure:

```text
unicorn-companies-data-validation/
│
├── README.md
├── Activity_Validate_and_Clean_Your_Data.ipynb
├── Modified_Unicorn_Companies.csv
│
└── requirements.txt
```

A simple `requirements.txt` can contain:

```text
numpy
pandas
matplotlib
seaborn
plotly
jupyter
```

---

# 📌 Project Status

**Status:** Completed

The notebook contains the completed data-validation and categorical-encoding workflow, including:

* Data inspection
* Data type conversion
* Feature engineering
* Invalid-data correction
* Industry-label standardization
* Duplicate removal
* Valuation categorization
* Ordinal encoding
* Nominal label encoding
* Dummy/one-hot encoding
* Final reflection and conclusions

---

# 🎓 Learning Context

This project was completed as part of the **Google Advanced Data Analytics Professional Certificate**, Course 2.

It represents a practical exercise in preparing real-world-style business data for analysis.

The activity uses an investment-firm scenario where the quality and structure of unicorn-company data are important for identifying meaningful patterns and supporting potential investment decisions.

---

# 📖 Reference

Bhat, M.A. — *Unicorn Companies*

[Kaggle Dataset](https://www.kaggle.com/datasets/mysarahmadbhat/unicorn-companies)

---

# ⚠️ Disclaimer

This repository is an educational project created for learning and portfolio purposes.

The analysis should not be interpreted as financial or investment advice.

The dataset and exercise scenario are used for educational data-analysis practice.

---

## 👤 Author

**Heinn Htet Zan**

Computer Science Student | Data Analytics & Machine Learning Enthusiast

* GitHub: https://github.com/Koheinn
* LinkedIn: https://www.linkedin.com/in/heinn-htet-zan-040794291/
* Portfolio: https://hhzportfolio.netlify.app

---

⭐ If you find this project useful, feel free to explore the notebook and the data-cleaning workflow.
