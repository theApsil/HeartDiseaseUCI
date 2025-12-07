import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/heart_disease_uci.csv')

summary = df.describe(include='all')
print(summary)

plt.figure()
df['num'].value_counts().sort_index().plot(kind='bar')
plt.title('Распределение целевой метки (num)')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.tight_layout()
plt.show()
