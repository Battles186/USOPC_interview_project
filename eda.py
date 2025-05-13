"""
Exploratory data analysis for load and wellness data.
"""

# Data management.
import numpy as np
import pandas as pd

# Visualization.
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities.
from util import preprocess_data, pos_group_colors, var_groups

# Statistics.
from scipy.stats import shapiro, boxcox

# Load data.
df = pd.read_excel('LoadandWellnessData.xlsx')
df_raw = df.copy()

# Preprocess.
df = preprocess_data(df)

# Assess the sparsity of the data.
df_nan_count = df.isna().sum(axis=0)/df.shape[0]
sns.set_color_codes('pastel')
sns.barplot(x=df_nan_count.values, y=df_nan_count.index)
plt.xlabel('Proportion of Missing Values')
plt.ylabel('')
plt.tight_layout()
plt.savefig('images/eda/eda_prop_missing.png')
plt.close()

# Histograms for each numerical variable.
df_num = df.select_dtypes(include=np.number)
fig, axs = plt.subplots(nrows=4, ncols=4)
_ = [
    (
        sns.histplot(data=df_raw, x=col, ax=ax),
        # ax.set_title(col)
        ax.set_ylabel('')
    )
    for col, ax in zip(df_num.columns, axs.ravel())
]
plt.tight_layout(pad=0.2)
plt.savefig('images/eda/eda_hist_all.png')
plt.close()

# Compare athletes by position group.
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
_ = [
    (
        ax.hist(
            df[df['Position'] == pos_group][col],
            color=pos_group_colors[pos_group],
            alpha=0.5,
        ),
        ax.set_xlabel(col),
    )
    for col, ax in zip(df_num.columns, axs.ravel())
    for pos_group in df['Position'].unique()
]
plt.tight_layout(pad=0.2)
plt.savefig('images/eda/eda_hist_all_pos_group.png')
plt.close()

# Look at means across the four groups throughout the entire dataset.
_ = [
    (
        df.groupby('Position')[vars_].mean().T.plot(
            kind='bar',
        ),
        plt.tight_layout(pad=0.2),
        plt.savefig(f'images/eda/eda_bar_pos_group_{var_group}'),
        plt.close(),
    )
    for var_group, vars_ in var_groups.items()
]

# Look at trends across time in each of the variables.
# _ = [
#     (
#         df.groupby('Position').plot(
#             'Date',
#             col
#         ),
#         plt.savefig(f'images/eda/time_series_pos_group_{col}.png'),
#         plt.close(),
#     )
#     for col in df_num.columns
# ]

# Show distributions of all the variables in our data with
# respect to position group.
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
_ = [
    (
        sns.boxplot(
            data=df,
            x='Position',
            hue='Position',
            y=col,
            ax=ax,
            legend='brief',
        ),
        # print(ax.get_xticklabels()),
        ax.set_xticks([], labels=[]),
        ax.set_xlabel(''),
    )
    for col, ax in zip(df_num.columns, axs.ravel())
]
plt.tight_layout(pad=0.2)

handles, labels = axs.ravel()[0].get_legend_handles_labels()
_ = [
    ax.get_legend().remove()
    for ax in axs.ravel()
]
plt.legend(
    handles=handles,
    labels=labels,
    loc='lower center',
    shadow=True,
    # bbox_to_anchor=(0.025, 0.025, 0.975, 0.975),
    # ncol=len(pos_group_colors),
)

# Thank you
# https://stackoverflow.com/questions/61980929/moving-and-removing-legend-from-seaborn-lineplot
# plt.figlegend(loc='lower right', bbox_to_anchor=(0.85, 0.25))

plt.savefig('images/eda/eda_box_all_pos_group.png')
plt.close()

# # Plot the trajectory of each position group as a function of time.
# 
# sns.lineplot(
#     data=df,
#     x='Date',
#     y='Stress',
#     hue='Position'
# )

print(f'n unique days: {len(df.Date.unique())}')
print(f'n unique athletes: {len(df.Athlete.unique())}')

# Assess multicollinearity.
df_num_corr = df_num.corr()
df_num_corr_flat = df_num_corr.unstack()
print("The following variables are correlated at or above 0.7:")
print(df_num_corr_flat[df_num_corr_flat > 0.7])

# Assess normality.
_ = [
    print(f"Shapiro-Wilk p value for {outcome}: {shapiro(df[outcome].dropna()).pvalue}")
    for outcome in ['Sleep', 'Stress', 'Fatigue', 'Motivation', 'Mood']
]

