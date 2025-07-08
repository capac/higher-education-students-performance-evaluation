#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


work_dir = Path.home() / (
    'Programming/Python/machine-learning-exercises/'
    'higher-education-students-performance-evaluation'
    )
data_file = work_dir / 'data/students-performance.csv'
attribute_names_json_file = work_dir / 'attribute_names.json'
path_dir = work_dir / 'plots'
path_dir.mkdir(parents=True, exist_ok=True)

with open(attribute_names_json_file, 'rt') as f_in:
    attribute_names_json = json.load(f_in)

sp_df = pd.read_csv(data_file)
column_names = list(sp_df.columns[1:-2])
column_counts = []
for col in column_names:
    column_counts.append(sp_df.groupby(col, observed=False).size())

fig, axs = plt.subplots(nrows=14, ncols=2, figsize=(13, 45))
for ax, col_count, col_name in zip(axs.flatten(), column_counts, column_names):
    if col_name in ['15', '22', '30']:
        ax.bar(list(attribute_names_json[col_name]['options'].values())[:-1],
               col_count.values)
    else:
        ax.bar(list(attribute_names_json[col_name]['options'].values()),
               col_count.values)
    ax.set_ylabel('Counts', fontsize=10)
    ax.set_title(attribute_names_json[col_name]['name'], fontsize=11)
    ax.yaxis.set_tick_params(pad=3)
    plt.tight_layout(pad=1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plot_file = path_dir / 'attribute_bar_plots.png'
    plt.savefig(plot_file, dpi=144, bbox_inches='tight')


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
grouped = sp_df.groupby('GRADE').size()
ax.bar(attribute_names_json['32']['options'].values(), grouped.values)
ax.set_ylabel('Counts', fontsize=10)
ax.set_title(attribute_names_json['32']['name'].title(), fontsize=11)
ax.yaxis.set_tick_params(pad=3)
plt.tight_layout(pad=1)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plot_file = path_dir / 'grade_barplot.png'
plt.savefig(plot_file, dpi=144, bbox_inches='tight')
