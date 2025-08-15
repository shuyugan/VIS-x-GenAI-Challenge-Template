import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./dataset.csv")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

# Filter out null values if exclude_null is True
df_filtered = df[df['GraphicsReplicabilityStamp'].notnull()]

# Count the occurrences of each category
stamp_counts = df_filtered['GraphicsReplicabilityStamp'].value_counts()

# Plot the pie chart
plt.pie(stamp_counts, labels=stamp_counts.index, colors=["#FF9999", "#66B3FF"], autopct='%1.1f%%', startangle=140)

plt.title('Proportion of Papers with Graphics Replicability Stamp')
plt.tight_layout()
plt.savefig("plot_10.png", dpi=300)
plt.close()