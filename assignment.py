# ================================
# Import Libraries
# ================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ================================
# Task 1: Load and Explore Dataset
# ================================

try:
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species"] = df["species"].map({
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    })
except Exception as e:
    print("Error loading dataset:", e)

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check dataset structure
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# ================================
# Data Cleaning
# ================================
# Iris dataset has no missing values,
# but this is included for completeness
df = df.dropna()

# ================================
# Task 2: Basic Data Analysis
# ================================

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and calculate mean
grouped_means = df.groupby("species").mean()
print("\nMean values grouped by species:")
print(grouped_means)

# ================================
# Task 3: Data Visualization
# ================================

# 1️⃣ Line Chart (Trend example)
plt.figure()
plt.plot(df.index, df["sepal length (cm)"])
plt.title("Sepal Length Trend Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.show()

# 2️⃣ Bar Chart (Average petal length per species)
plt.figure()
grouped_means["petal length (cm)"].plot(kind="bar")
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3️⃣ Histogram (Distribution of sepal length)
plt.figure()
plt.hist(df["sepal length (cm)"], bins=15)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4️⃣ Scatter Plot (Sepal length vs Petal length)
plt.figure()
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species"
)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# ================================
# Observations (Printed)
# ================================
print("\nFindings & Observations:")
print("- Virginica has the largest average petal length and width.")
print("- Setosa has noticeably smaller petal measurements.")
print("- Petal features separate species more clearly than sepal features.")
