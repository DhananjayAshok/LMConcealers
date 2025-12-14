import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
default_plt_params = plt.rcParams.copy()

font_size = 24
labels_font_size = 28
xtick_font_size = 26
ytick_font_size = 26
title_font_size = 18
scaler = 1.0

# set font to times new roman
plt.rcParams.update({'font.family': 'Times New Roman'})
plt.rcParams.update({'font.size': font_size * scaler})
if labels_font_size is None:
    labels_font_size = default_plt_params["axes.labelsize"]
plt.rcParams.update({'axes.labelsize': labels_font_size * scaler})
if xtick_font_size is None:
    xtick_font_size = default_plt_params["xtick.labelsize"]
plt.rcParams.update({'xtick.labelsize': xtick_font_size * scaler})
if ytick_font_size is None:
    ytick_font_size = default_plt_params["ytick.labelsize"]
plt.rcParams.update({'ytick.labelsize': ytick_font_size * scaler})
if title_font_size is None:
    title_font_size = default_plt_params["axes.titlesize"]
plt.rcParams.update({'axes.titlesize': title_font_size * scaler})

df = pd.read_csv("plot_data.csv", index_col="Model")
# plot heatmap
cbar_kws={'label': 'Detection Accuracy'}
#cbar_kws={}
sns.heatmap(df, annot=True, fmt=".0f", center=60, cmap="coolwarm_r", cbar_kws=cbar_kws, linecolor="black", linewidths=0.5) 
plt.title("")
# tilt the x labels
plt.xticks(rotation=20, ha="right")
# tilt the y labels
plt.yticks(rotation=75)
plt.xlabel("Topic")
plt.ylabel("Model")
plt.tight_layout()
plt.show()