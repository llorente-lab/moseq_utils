"""
Scripts for plotting syllable usage stats
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import warnings
from scipy import stats
import os
from scikit_posthocs import posthoc_dunn
import matplotlib as mpl
from scipy.spatial.distance import jensenshannon
from config import MAPPING_MALES, COLOR_MAPPING_MALES, MAPPING_FEMALES, COLOR_MAPPING_FEMALES

def _get_mapping(mapping):
    if mapping.lower() == 'male':
        return MAPPING_MALES, COLOR_MAPPING_MALES
    elif mapping.lower() == 'female':
        return MAPPING_FEMALES, COLOR_MAPPING_FEMALES
    else:
        raise ValueError(f'Mapping must be male or female, not {mapping}')

def run_kw_dunn(data, group_col, value_col, control_group):
    """
    Run Kruskal-Wallis H-test and Dunn's post-hoc test to identify significant differences.
    
    Args:
    data (pandas.DataFrame): DataFrame containing the data
    group_col (str): Name of the column containing group labels
    value_col (str): Name of the column containing the values to compare
    control_group (str): Name of the control group
    
    Returns:
    list: List of syllables with significant differences compared to the control group
    """
    groups = data[group_col].unique()
    syllables = data['syllable'].unique()
    significant_sylls = []
    
    for syll in syllables:
        syll_data = data[data['syllable'] == syll]
        group_data = [syll_data[syll_data[group_col] == group][value_col] for group in groups]
        
        # Kruskal-Wallis H-test
        h_statistic, p_value = stats.kruskal(*group_data)
        
        if p_value < 0.01:  # If significant difference exists
            # Dunn's post-hoc test
            posthoc = posthoc_dunn(syll_data, val_col=value_col, group_col=group_col, p_adjust='bonferroni')
            
            # Check if any group is significantly different from the control group
            control_index = list(groups).index(control_group)
            if any(posthoc.iloc[control_index, :] < 0.01):
                significant_sylls.append(syll)
    
    return significant_sylls

def plot_syllable_usage_comparison(stats_df, group1_name, group2_name, mapping, syll_info=None, max_sylls=40, colors=None, figsize=(10, 6)):
    """
    Plot a comparison of syllable usage between two groups with error bars and significance markers.
    
    Args:
    stats_df (pandas.DataFrame): dataframe containing the summary statistics about syllable data
    group1_name (str): Name of the first group to compare (control group)
    group2_name (str): Name of the second group to compare (experimental group)
    mapping (str): sex of the specified group (male or female)
    syll_info (dict): dictionary of syllable numbers mapped to dict containing the label, description and crowd movie path
    max_sylls (int): the maximum number of syllables to include
    colors (list): list of user-selected colors to represent the data
    figsize (tuple): tuple value representing (width, height) of the plotted figure dimensions
    
    Returns:
    fig (pyplot.figure): plotted syllable usage comparison
    legend (pyplot.legend): figure legend
    """

    try:
       MAPPING, COLOR_MAPPING = _get_mapping(mapping)
    except ValueError as e:
        print(f"Error: {e}")
        return None, None, None # lets keep the return type consistent
        
    # Validate and prepare data
    stats_df = stats_df[stats_df['syllable'] < max_sylls]
    stats_df['group'] = stats_df['group'].replace(MAPPING)
    groups = [group1_name, group2_name]
  
    colors = [COLOR_MAPPING[group1_name], COLOR_MAPPING[group2_name]]
    
    # Run statistical tests to identify significant syllables
    sig_sylls = run_kw_dunn(stats_df, 'group', 'usage', group1_name)
    
    # Calculate differences and sort syllables
    group1 = stats_df[stats_df['group'] == group1_name]
    group2 = stats_df[stats_df['group'] == group2_name]
    syllables = pd.Index(set(group1['syllable'].unique()) | set(group2['syllable'].unique()))
    diff = group2.groupby('syllable')['usage'].mean().reindex(syllables).fillna(0) - \
           group1.groupby('syllable')['usage'].mean().reindex(syllables).fillna(0)
    ordering = diff.sort_values(ascending=False).index
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    fig.dpi = 200
    
    # Plot each group's usage data separately
    for i, group in enumerate([group1_name, group2_name]):
        group_data = stats_df[stats_df['group'] == group].copy()
        group_data['usage'] *= 100
        sns.pointplot(data=group_data, x='syllable', y='usage', order=ordering,
                      join=False, dodge=True, errorbar=('ci', 34), ax=ax, color=colors[i])
    
    # Customize the plot
    ax.set_xlabel('Syllable ID', fontsize=12)
    ax.set_ylabel('Syllable usage (%)', fontsize=12)
    ax.set_title(f'Syllable Usage Comparison: {group1_name} vs {group2_name}', fontsize=14)
    
    # Add syllable labels if they exist
    if syll_info is not None:
        mean_xlabels = [f'{syll_info[o]["label"]} - {o}' for o in ordering]
        plt.xticks(range(len(ordering)), mean_xlabels, rotation=90)
    
    # Mark significant syllables
    markings = [ordering.get_loc(s) for s in sig_sylls if s in ordering]
    plt.scatter(markings, [ax.get_ylim()[1]] * len(markings), color='r', marker='*', s=100)
    
    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', 
                             markersize=9, label=group) for color, group in zip(colors, groups)]
    handles.append(mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                             markersize=9, label="Significant Syllable"))
    legend = ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.05, 1))
    
    # Remove top and right spines
    sns.despine()
    
    return fig, legend, group_data

def jensen_shannon_divergence(stats_df, mapping, groups=['all'], save_path='jsd_matrix.png'):
    """
    Compute Jensen-Shannon divergence between a pair of experimental groups
    
    Args:
    stats_df (pandas.DataFrame): dataframe containing the summary statistics about syllable data
    groups (list): groups to compute Jensen-Shannon divergence between
    save_path (str): name of the file to save it to
    
    Returns:
    jsd_matrix (numpy.array): NumPy array of shape (x, x) where x represents the number of groups for whom JSD is being computed for
    fig (pyplot.figure): plotted syllable usage comparison
    """

    assert not stats_df.empty, "The stats_df dataframe is empty"
    stats_df['group'] = stats_df['group'].replace(MAPPING)

    save_path = os.path.join(os.getcwd(), save_path) # convert absolute path to relative path

    if groups != ['all']:
        # Handle case where we aren't computing jensen shannon divergence for all groups
        stats_df = stats_df[stats_df['group'].isin(groups)]

    grouped_df = stats_df.groupby(['group', 'syllable'])['usage'].mean().reset_index()
    grouped_df['probability'] = grouped_df.groupby('group')['usage'].transform(lambda x: x / x.sum())
    pivot_df = grouped_df.pivot(index='syllable', columns='group', values='probability').fillna(0)

    groups = pivot_df.columns
    jsd_matrix = np.zeros((len(groups), len(groups)))
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:  # Compute only for upper triangle
                jsd = jensenshannon(pivot_df[group1], pivot_df[group2])
                jsd_matrix[i, j] = jsd
                jsd_matrix[j, i] = jsd  # JSD is symmetric
    # create matplotlib fig
    fig, ax1 = plt.subplots(figsize=(6, 5))
    fig.dpi = 200

    sns.heatmap(jsd_matrix, ax=ax1, annot=False, cmap='viridis_r', xticklabels=groups, yticklabels=groups, cbar_kws={'label': 'Jensen-Shannon Divergence'})
    ax1.set_title('Jensen-Shannon Divergence between Group Syllable Usage Distributions', fontsize=16)
    ax1.set_xlabel('Group', fontsize=12)
    ax1.set_ylabel('Group', fontsize=12)

    # rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

    return jsd_matrix, fig
