# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:28:28 2024

@author: huber
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path

from file_utils import create_directory, check_folder_exists


#%%
def create_multi_series_scatter_plot_plotly(data, **kwargs):
    """
    Creates a scatter plot with multiple data series using Plotly.

    Parameters:
    - data: List of dictionaries, where each dictionary represents a dataset to plot.
            Each dictionary should have 'x' and 'y' keys for the data points.
    - kwargs: Additional keyword arguments for customization.

    Returns:
    - fig: Plotly figure with the created scatter plot.
    """
    # Extract additional keyword arguments
    legend_labels = kwargs.get('legend_labels', [])
    scatter_colors = kwargs.get('scatter_colors', [])
    plot_title = kwargs.get('plot_title', 'Multi-Series Scatter Plot')
    x_label = kwargs.get('x_label', 'X-axis')
    y_label = kwargs.get('y_label', 'Y-axis')
    show_grid = kwargs.get('show_grid', False)

    # Initialize the figure
    fig = go.Figure()

    # Add each series as a scatter trace
    for i, series in enumerate(data):
        name = legend_labels[i] if i < len(legend_labels) else f'Series {i+1}'
        color = scatter_colors[i] if i < len(scatter_colors) else None
        
        fig.add_trace(go.Scatter(
            x=series.get('x', []),
            y=series.get('y', []),
            mode='markers',
            name=name,
            marker=dict(color=color)
        ))

    # Update plot layout
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=True,
        xaxis=dict(showgrid=show_grid),
        yaxis=dict(showgrid=show_grid)
    )

    return fig

#%%
def create_subplots_plotly(n_plots, n_cols=2, figsize=(30, 5)):
    """
    Creates a figure with subplots in a grid layout.

    Parameters:
    - n_plots (int): The number of subplots to create.
    - n_cols (int, optional): The number of columns in the subplot grid. Default is 2.
    - figsize (tuple, optional): The size of the figure in inches (width, height). Default is (30, 5).
    
    Returns:
    - fig (plotly.graph_objects.Figure): The created Plotly figure object with subplots.
    """
    # Validate inputs
    if n_plots < 1:
        raise ValueError("The number of plots (n_plots) must be at least 1.")
    
    if n_cols < 1:
        raise ValueError("The number of columns (n_cols) must be at least 1.")

    # Determine the number of rows required to fit the plots
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create the figure with subplots
    fig = make_subplots(rows=n_rows, cols=n_cols)

    # Adjust the size of the figure
    fig.update_layout(height=figsize[1] * 100, width=figsize[0] * 100)

    return fig

#%%
def save_html_plotly(fig, file_path: str) -> None:
    """
    Save a Plotly plot to a file in the specified format and directory.
    """
    supported_formats = ["html"]

    # Extract file extension
    file_extension = Path(file_path).suffix.lstrip('.')

    # Ensure valid format
    if file_extension not in supported_formats:
        raise ValueError(f"Unsupported format '{file_extension}'. Supported formats are: {', '.join(supported_formats)}.")

    # Ensure directory exists
    dir_path = Path(file_path).parent
    if(check_folder_exists(dir_path)):
        create_directory(dir_path)

    # Check if fig is a Plotly figure
    if isinstance(fig, go.Figure):
        if file_extension == "html":
            pio.write_html(fig, file_path)
            print("Figure saved successfully!")
    else:
        raise TypeError("The 'fig' parameter must be a Plotly 'go.Figure'.")

#%%



#%%
if __name__ == "__main__":
    current_working_directory = Path.cwd()
    output_file_path = current_working_directory / 'plots' / 'create_multi_series_scatter_plot_plotly.html'
    
    data = [
        {'x': [1, 2, 3], 'y': [4, 5, 6]},
        {'x': [1, 2, 3], 'y': [7, 8, 9]},
        {'x': [1, 2, 3], 'y': [10, 11, 12]}
    ]
    
    fig = create_multi_series_scatter_plot_plotly(
        data,
        legend_labels=['Series 1', 'Series 2', 'Series 3'],
        scatter_colors=['red', 'blue', 'green'],
        plot_title='Example Multi-Series Scatter Plot',
        x_label='Time [ms]',
        y_label='Value',
        show_grid=True
    )
    
    save_html_plotly(fig, output_file_path)
