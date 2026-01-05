import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("sales_data.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Calculate total revenue
total_revenue = df["Revenue ($)"].sum()

# Find best-selling product (by Quantity Sold)
best_selling = (
    df.groupby("Product")["Quantity Sold"]
    .sum()
    .idxmax()
)

best_selling_qty = (
    df.groupby("Product")["Quantity Sold"]
    .sum()
    .max()
)

# Identify the day with the highest sales (by Revenue)
highest_sales_day = (
    df.groupby("Date")["Revenue ($)"]
    .sum()
    .idxmax()
)

# Prepare summary text
summary = f"""
Total Revenue: ${total_revenue:,.0f}
Best-Selling Product: {best_selling} ({best_selling_qty} units sold)
Highest Sales Day: {highest_sales_day.date()}
"""

# Save summary to text file
with open("sales_summary.txt", "w") as file:
    file.write(summary.strip())

# Print insights
print(" SALES INSIGHTS")
print("------------------")
print(summary)

# Visualize sales trends
daily_sales = df.groupby("Date")["Revenue ($)"].sum()

plt.figure()
plt.plot(daily_sales.index, daily_sales.values, marker="o")
plt.xlabel("Date")
plt.ylabel("Revenue ($)")
plt.title("Daily Sales Revenue Trend")
plt.show()
