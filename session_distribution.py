import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
from collections import Counter
import seaborn as sns

df = pd.read_csv('Demographics.csv')
df['BIDS'] = df['BIDS'].astype(str).str.strip()
bids_counts_series = df['BIDS'].value_counts()
bids_counts_df = bids_counts_series.reset_index()
bids_counts_df.columns = ['BIDS', 'Sessions']

session_counts = bids_counts_df['Sessions'].value_counts().sort_index()
for i in range(1, 8):
    if i not in session_counts:
        session_counts.loc[i] = 0

# Sort the session counts by session number (index)
session_counts = session_counts.sort_index()

plt.figure(figsize=(12, 8))
bars = plt.bar(session_counts.index, session_counts.values, color='skyblue')

# Add labels for the count of people on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Set axis labels and title
plt.xlabel('Number of Sessions', fontsize=24)
plt.ylabel('Number of Patients', fontsize=24)
plt.title('Distribution of Number of Sessions', fontsize=30)

# Customize ticks, labels, and appearance
plt.xticks(range(1, 8))  # Ensure the x-axis has labels for sessions 1 through 7
#plt.xticklabels([str(i) for i in range(1, 8)], fontsize=12)
plt.tick_params(axis='y', labelsize=12)

# Tight layout for better spacing
plt.tight_layout()

# Display the plot
#plt.show()
plt.savefig('session_distribution.png')
