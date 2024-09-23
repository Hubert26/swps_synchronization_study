# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:28:28 2024

@author: huber
"""


import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from src.utils.file_utils import create_directory

#%%
def create_bar_plot(data, column, ax, title='', xlabel='', ylabel='Count', color='blue'):
    """
    Creates a bar plot for the specified column.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data to plot.
    - column (str): The column name for which the distribution will be plotted.
    - ax (matplotlib.axes.Axes): The matplotlib Axes object where the plot will be drawn.
    - title (str, optional): The title for the plot. Default is an empty string.
    - xlabel (str, optional): The label for the x-axis. Default is an empty string.
    - ylabel (str, optional): The label for the y-axis. Default is 'Count'.
    - color (str, optional): The color of the bars in the plot. Default is 'blue'.
    
    Raises:
    - ValueError: If the specified column is not present in the DataFrame.
    - TypeError: If the provided axis (`ax`) is not a valid matplotlib Axes object.
    """
    # Validate inputs
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the provided DataFrame.")
    
    if not isinstance(ax, plt.Axes):
        raise TypeError("The provided ax is not a valid matplotlib Axes object.")
    
    # Calculate the value counts for the column
    distribution = data[column].value_counts()

    # Create the bar plot
    distribution.plot(kind='bar', ax=ax, color=color)

    # Set plot titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel(ylabel)

#%%
def create_multi_series_bar_chart_matplotlib(data, ax=None, fill_missing: bool = True, **kwargs):
    """
    Creates a bar plot with multiple data series using an existing Matplotlib axis (for use in subplots).
    
    Parameters:
    - data: List of dictionaries, where each dictionary represents a dataset to plot.
            The keys of the dictionaries are used as the x-axis labels.
    - ax: Matplotlib axis object to plot on. If None, creates a new axis.
    - fill_missing: Boolean, whether to fill missing keys with 0 (default is True).
    - kwargs: Additional keyword arguments for customization.

    Additional Keyword Arguments:
    - legend_labels: List of labels for the legend.
    - bar_colors: List of colors for the bars.
    - plot_title: Title of the plot.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - bar_width: Width of the bars (default is 0.35).
    - legend_position: Position of the legend (default is 'best').
    - show_grid: Boolean to show gridlines (default is False).
    - horizontal: Whether to plot bars horizontally (default is False).
    - show_values: Boolean to display the values above the bars (default is False).
    - value_format: Format string to customize how the values are displayed above the bars (default is "{:.1f}").
    - x_label_rotation: Angle (in degrees) for rotating x-axis category labels (default is 0).
    - value_rotation: Angle (in degrees) for rotating the values displayed above the bars (default is 0).
    
    Returns:
    - ax: Matplotlib axis with the created bar plot.
    """

    # Extract all unique keys from the data to use as labels for the x-axis
    labels = sorted(set().union(*(d.keys() for d in data)))

    # Retrieve keyword arguments or use defaults
    legend_labels = kwargs.get('legend_labels', [f'Series {i+1}' for i in range(len(data))])
    bar_colors = kwargs.get('bar_colors', plt.cm.tab10.colors)
    plot_title = kwargs.get('plot_title', 'Multiple Bar Plot')
    x_label = kwargs.get('x_label', 'Categories')
    y_label = kwargs.get('y_label', 'Values')
    bar_width = kwargs.get('bar_width', 0.35)
    legend_position = kwargs.get('legend_position', 'best')
    show_grid = kwargs.get('show_grid', False)
    horizontal = kwargs.get('horizontal', False)
    show_values = kwargs.get('show_values', False)
    value_format = kwargs.get('value_format', "{:.1f}")
    x_label_rotation = kwargs.get('x_label_rotation', 0)
    value_rotation = kwargs.get('value_rotation', 0)

    # Prepare the data for plotting, filling in missing keys if necessary
    plot_data = []
    for d in data:
        if fill_missing:
            # Fill missing keys with 0 if the key does not exist
            plot_data.append([d.get(label, 0) for label in labels])
        else:
            # Use None or skip missing keys (optional, could use another approach)
            plot_data.append([d[label] if label in d else None for label in labels])

    # If no axis is provided, create one
    if ax is None:
        fig, ax = plt.subplots()

    # Calculate positions for the bars
    num_categories = len(labels)
    total_width = bar_width * len(data)
    spacing = (1 - total_width) / (num_categories + 1)  # Space between bars

    x = np.arange(num_categories) * (total_width + spacing)  # Space between categories

    for i, series in enumerate(plot_data):
        if horizontal:
            bars = ax.barh(x + i * bar_width, series, bar_width, label=legend_labels[i], color=bar_colors[i])
        else:
            bars = ax.bar(x + i * bar_width, series, bar_width, label=legend_labels[i], color=bar_colors[i])

        # Add values on top of the bars
        if show_values:
            for bar in bars:
                value = bar.get_width() if horizontal else bar.get_height()
                if value != 0:  # Only show values for non-zero bars
                    if horizontal:
                        ax.text(value, bar.get_y() + bar.get_height() / 2,
                                value_format.format(value), va='center', ha='left', rotation=value_rotation)
                    else:
                        ax.text(bar.get_x() + bar.get_width() / 2, value,
                                value_format.format(value), ha='center', va='bottom', rotation=value_rotation)
                        
    # Customize the plot
    ax.set_title(plot_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x + bar_width * (len(data) - 1) / 2)
    ax.set_xticklabels(labels, rotation=x_label_rotation)  # Rotate x-axis labels
    if show_grid:
        ax.grid(True)
    ax.legend(loc=legend_position)

    return ax

#%%
def create_subplots_matplotlib(n_plots, n_cols=2, figsize=(30, 5)):
    """
    Creates a figure with subplots in a grid layout.

    Parameters:
    - n_plots (int): The number of subplots to create.
    - n_cols (int, optional): The number of columns in the subplot grid. Default is 2.
    - figsize (tuple, optional): The size of the figure in inches (width, height). Default is (15, 5).
    
    Returns:
    - fig (matplotlib.figure.Figure): The created matplotlib figure object.
    - axes (list of matplotlib.axes.Axes): A flattened list of axes objects (subplots).
    
    Raises:
    - ValueError: If the number of plots (`n_plots`) is less than 1 or if `n_cols` is less than 1.
    """
    # Validate inputs
    if n_plots < 1:
        raise ValueError("The number of plots (n_plots) must be at least 1.")
    
    if n_cols < 1:
        raise ValueError("The number of columns (n_cols) must be at least 1.")

    # Determine the number of rows required to fit the plots
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array for easy iteration
    axes = axes.flatten() if n_plots > 1 else [axes]

    return fig, axes

#%%
def save_fig_matplotlib(fig, file_path: str) -> None:
    """
    Save a Matplotlib or Seaborn plot to a file in the specified format and directory.

    This function saves a plot to a file with the format specified in the file_path extension 
    and ensures that the output directory exists. It supports both Matplotlib and Seaborn figures.

    Parameters:
    - fig (plt.Figure): The plot object to be saved. Can be a Matplotlib or Seaborn figure.
    - file_path (str): The path where the plot will be saved, including the file name and extension.

    Raises:
    - ValueError: If the file format (extracted from file_path) is not supported.
    - TypeError: If the provided figure is neither a Matplotlib nor a Seaborn figure.

    Returns:
    None
    """

    # List of supported formats
    supported_formats = ["png", "jpg", "svg", "pdf"]

    # Extract the format from the file path
    file_extension = Path(file_path).suffix.lstrip('.')
    
    # Ensure the provided format is valid
    if file_extension not in supported_formats:
        raise ValueError(f"Unsupported format '{file_extension}'. Supported formats are: {', '.join(supported_formats)}.")
    
    # Ensure the directory exists
    dir_path = Path(file_path).parent
    if dir_path.is_dir():
        create_directory(dir_path)
    
    # Check if the figure is a Matplotlib or Seaborn figure
    if isinstance(fig, plt.Figure):
        # Save the Matplotlib or Seaborn figure as an image file (PNG, JPG, SVG, PDF)
        fig.savefig(file_path, format=file_extension)
    else:
        raise TypeError("The 'fig' parameter must be a Matplotlib 'plt.Figure'.")
        
#%%
# =============================================================================
# def corr_heatmap(df, title=None, color='viridis'):
#     # Tworzenie własnej mapy kolorów z 20 odcieniami od -1 do 1
#     colors = sns.color_palette(color, 20)
#     cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=20)
#     
#     with sns.axes_style("white"):
#         f, ax = plt.subplots(figsize=(10, 10))
#         sns.heatmap(df,
# # =============================================================================
# # to annotate on heatmap you need previous version of matplotlib              
# # pip install matplotlib==3.7.3
# # =============================================================================
#                     annot=df.round(2),
#                     vmax=1,
#                     vmin=-1,
#                     center=0,
#                     square=True,
#                     xticklabels=df.columns,
#                     yticklabels=df.index,
#                     cmap=cmap,
#                     linewidths=.5,
#                     cbar_kws={"shrink": 0.7, 'ticks': np.linspace(-1, 1, 21)})
#         # Ustawienie rotacji etykiet
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#     
#     if not title:
#         title = 'heatmap'
#     
#     plt.title(title)
# 
#     return f, title
# =============================================================================


#%%
if __name__ == "__main__":
    current_working_directory = Path.cwd()
    output_file_path = current_working_directory / 'plots' / 'subplots_multi_series_bar_charts.png'
    
    # Sample data: list of dictionaries
    data1 = [{'A': 5, 'B': 3, 'C': 7}, {'A': 2, 'C': 4, 'D': 5}, {'B': 6, 'C': 3, 'D': 4}]
    data2 = [{'A': 4, 'B': 2, 'C': 5}, {'A': 1, 'B': 4, 'C': 6}]
    data3 = [{'X': 3, 'Y': 5, 'Z': 6}, {'X': 4, 'Y': 7}]
    data4 = [{'P': 6, 'Q': 8, 'R': 4}, {'P': 5, 'Q': 7}]

    # Create subplots: 4 plots in 2 columns
    fig, axes = create_subplots_matplotlib(n_plots=4, n_cols=2, figsize=(12, 8))

    # Use create_multiple_bar_plot to create plots in each subplot
    create_multi_series_bar_chart_matplotlib(data1, ax=axes[0],
                             plot_title="Plot 1",
                             legend_labels=["Series 1", "Series 2", "Series 3"],
                             bar_width = 0.20,
                             show_values = True,
                             x_label_rotation = 45)
    
    create_multi_series_bar_chart_matplotlib(data2,
                             ax=axes[1],
                             plot_title="Plot 2",
                             legend_labels=["Series 1", "Series 2"],
                             bar_width = 0.20,
                             show_values = True,
                             x_label_rotation = 45)
    
    create_multi_series_bar_chart_matplotlib(data3,
                             ax=axes[2],
                             plot_title="Plot 3",
                             legend_labels=["Series 1", "Series 2"],
                             bar_width = 0.20,
                             show_values = True,
                             x_label_rotation = 45)
    
    create_multi_series_bar_chart_matplotlib(data4,
                             ax=axes[3],
                             plot_title="Plot 4",
                             legend_labels=["Series 1", "Series 2"],
                             bar_width = 0.20,
                             show_values = True,
                             x_label_rotation = 45)

    # Adjust layout
    plt.tight_layout()

    # Save plots to file
    save_fig_matplotlib(fig, file_path=output_file_path)