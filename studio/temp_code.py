import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./dataset.csv")

internal_references = df['InternalReferences'].dropna().str.split(';')
edges = [(doi, ref) for doi, refs in zip(df['DOI'], internal_references) for ref in refs]

plt.figure(figsize=(12, 10))
plt.title('Internal References Network')
plt.axis('off')
plt.tight_layout()
plt.savefig("plot_10.png", dpi=300)
plt.close()