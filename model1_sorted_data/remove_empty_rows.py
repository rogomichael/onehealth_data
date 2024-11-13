import pandas as pd

# Load your data
df = pd.read_excel("m1train_poultry_filled.xlsx")  # or pd.read_excel("your_file.xlsx")

# Remove rows where all columns are empty
print("Removing rows")
df = df.dropna(how='any')

# Save the cleaned data
df.to_csv("m1train_poultry_filled.csv", index=False)