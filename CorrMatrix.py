import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/heart_disease_uci.csv')

num_cols = df.select_dtypes(include=['float64','int64']).columns
num_df = df[num_cols]

corr = num_df.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr)
plt.title("Correlation Matrix Heatmap")
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)
plt.colorbar()
plt.tight_layout()
plt.show()