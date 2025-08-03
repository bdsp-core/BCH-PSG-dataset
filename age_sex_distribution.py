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
df['Sex'] = df['Sex'].str.strip().str.upper()

# Define age bins and labels
bins = [0, 181, 366, 2191, 4381, 6571, float('inf')]
labels = ['<6mo', '6mo-1y', '1-6y', '6-12y', '12-18y', '>18y']

# Create a new column for age group
df['AgeGroup'] = pd.cut(df['AgeDays'], bins=bins, labels=labels, right=False)

# Initialize counters
males, females, undefined = [], [], []

# Loop through age groups and count by sex
for label in labels:
    group_df = df[df['AgeGroup'] == label]
    males.append((group_df['Sex'] == 'M').sum())
    females.append((group_df['Sex'] == 'F').sum())
    undefined.append((group_df['Sex'] == 'U').sum())

# Display the result
for i, group in enumerate(labels):
    print(f"{group}: M={males[i]}, F={females[i]}, U={undefined[i]}")
    
# Age group labels
age_groups = ['0-6 months', '6-12 months', '1-6 years', '6-12 years', '12-18 years', '>18 years']

fig, ax = plt.subplots(figsize=(10, 6))

# Position of the bars
x = np.arange(len(age_groups))

# Plotting
bars_f = ax.bar(x, females, color='lightpink', label='Female')
bars_u = ax.bar(x, undefined, bottom=females, color='purple', label='Unknown')
bars_m = ax.bar(x, males, bottom=np.array(females) + np.array(undefined), color='lightblue', label='Male')

# Annotate bars with counts (inside the bars)
for i in range(len(x)):
    # Total counts for this group
    total = females[i] + undefined[i] + males[i]
    
    # Place the total number on top of the bar
    ax.text(x[i], total + 1, str(total), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Place the number of females inside the pink bar
    ax.text(x[i], females[i] / 2, str(females[i]), ha='center', va='center', fontsize=10, color='black')
    
    if undefined[i]!=0:
        # Place the number of undefined inside the purple bar
        ax.text(x[i], females[i] + undefined[i] / 2, str(undefined[i]), ha='center', va='center', fontsize=10, color='black')

    # Place the number of males inside the blue bar
    ax.text(x[i], females[i] + undefined[i] + males[i] / 2, str(males[i]), ha='center', va='center', fontsize=10, color='black')

# Set x-axis ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(age_groups)

# Set labels and title
ax.set_ylabel('Number of PSGs',fontsize=24)
ax.set_xlabel('Age Group', fontsize=24)
ax.set_title('Distribution of PSGs by age and sex',fontsize=30)

# Add a legend
ax.legend()

# Customize appearance
plt.tight_layout()

# Show plot
#plt.show()
plt.savefig('sex_distribution.png')