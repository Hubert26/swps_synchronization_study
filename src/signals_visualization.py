# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:36:10 2024

@author: huber
"""

import pandas as pd


from config import *
import importlib
from functions import *
from utils.file_utils import list_file_paths, create_directory, delete_directory

                
#%%
if __name__ == '__main__':
    file_paths = list_file_paths(DATA_DIR)
    meas_list = load_data(file_paths)
    
#%%
    #Ploting oryginal signals
    gruped_meas = group_meas(meas_list, ["meas_type", "meas_number"])

    for key, group in gruped_meas.items():
        merged_meas = merge_grouped(group)
        
        meas_type, meas_number = key
        folder_path = PLOTS_DIR / meas_type / str(meas_number) / 'ORYGINAL_SIGNALS'
        
        for meas in merged_meas:
            file_name = str(meas) + str(meas.data.range()[0]) + ".html"
        
            create_directory(folder_path / 'HISTOGRAM')
            fig_hist, title_hist = density_plot([meas])
            save_html_plotly(fig_hist, folder_path  / 'HISTOGRAM' / file_name)
            
            create_directory(folder_path / 'SCATTER')
            fig_scatter, title_scatter = scatter_plot([meas])
            save_html_plotly(fig_scatter,  folder_path / 'SCATTER' / file_name)
            
#%%
    #Ploting pairs signals
    gruped_meas = group_meas(meas_list, ["meas_type", "meas_number"])

    for key, group in gruped_meas.items():
        
        merged_meas = merge_grouped(group)
        
        meas_type, meas_number = key
        folder_path = PLOTS_DIR / meas_type / str(meas_number) / 'ORYGINAL_PAIRS'
        
        pairs = find_pairs(merged_meas)
        
        for pair in pairs:
            plot_meas1, plot_meas2 = copy.deepcopy(pair)
            plot_histogram_pair((plot_meas1, plot_meas2), folder_path)
            plot_scatter_pair((plot_meas1, plot_meas2), folder_path)

#%%
    #Ploting filtered pairs signals
    gruped_meas = group_meas(meas_list, ["meas_type", "meas_number"])

    for key, group in gruped_meas.items():
        
        filtered_meas = filter_meas(group)
        merged_meas = merge_grouped(group)
        
        meas_type, meas_number = key
        folder_path = PLOTS_DIR / meas_type / str(meas_number) / 'FILTERED_PAIRS'
        
        pairs = find_pairs(merged_meas)
        
        for pair in pairs:
            plot_meas1, plot_meas2 = copy.deepcopy(pair)
            plot_histogram_pair((plot_meas1, plot_meas2), folder_path)
            plot_scatter_pair((plot_meas1, plot_meas2), folder_path)