import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./dataset.csv")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

awarded_papers = df[df['Award'].notnull()]
awarded_papers_count = awarded_papers.groupby('Year').size()

plt.plot(awarded_papers_count.index, awarded_papers_count.values, color='red')

plt.title('Number of Awarded Papers Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Awarded Papers')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("plot_8.png", dpi=300)
plt.close()