# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:36:10 2024

@author: huber
"""

import pandas as pd


from src.config import *
from src.config import intervals
import importlib
from src.functions import *
from src.utils.file_utils import list_file_paths, create_directory, delete_directory

#%%
def main():
    file_paths = list_file_paths(DATA_DIR)
    meas_list = load_data(file_paths)
    
    
    #Ploting oryginal signals
    gruped_meas = group_meas(meas_list, ["meas_type", "meas_number"])

    for key, group in gruped_meas.items():
        merged_meas = merge_meas(group)
        
        meas_type, meas_number = key
        folder_path = PLOTS_DIR / meas_type / str(meas_number) / 'MEAS'
        
        for meas in merged_meas:
            file_name = str(meas) + str(meas.data.range()[0]) + ".html"
        
            create_directory(folder_path / 'HISTOGRAM')
            fig_hist, title_hist = density_plot([meas])
            save_html_plotly(fig_hist, folder_path  / 'HISTOGRAM' / file_name)
            
            create_directory(folder_path / 'SCATTER')
            fig_scatter, title_scatter = scatter_plot([meas])
            save_html_plotly(fig_scatter,  folder_path / 'SCATTER' / file_name)


    #Ploting pairs signals
    gruped_meas = group_meas(meas_list, ["meas_type", "meas_number"])

    for key, group in gruped_meas.items():
        
        merged_meas = merge_meas(group)
        
        meas_type, meas_number = key
        folder_path = PLOTS_DIR / meas_type / str(meas_number) / 'PAIRS'
        
        pairs = find_pairs(merged_meas)
        
        for meas1, meas2 in pairs:
            time_align_pair(meas1, meas2)
            
            file_name = (
                str(meas1) + ';' + str(meas2) 
                + str((
                    min(meas1.data.range()[0][0], meas2.data.range()[0][0]), 
                    max(meas1.data.range()[0][1], meas2.data.range()[0][1])
                ))
                + ".html"
            )
            
            create_directory(folder_path / 'HISTOGRAM')
            fig_hist, title_hist = density_plot([meas1, meas2])
            save_html_plotly(fig_hist, folder_path  / 'HISTOGRAM' / file_name)
            
            create_directory(folder_path / 'SCATTER')
            fig_scatter, title_scatter = scatter_plot([meas1, meas2])
            save_html_plotly(fig_scatter,  folder_path / 'SCATTER' / file_name)


    #Ploting filtered pairs signals
    gruped_meas = group_meas(meas_list, ["meas_type", "meas_number"])

    for key, group in gruped_meas.items():
        
        filtered_meas = filter_meas(group)
        merged_meas = merge_meas(filtered_meas)
        
        meas_type, meas_number = key
        folder_path = PLOTS_DIR / meas_type / str(meas_number) / 'PAIRS_FILTERED'
        
        pairs = find_pairs(merged_meas)
        
        for meas1, meas2 in pairs:
            time_align_pair(meas1, meas2)
            
            file_name = (
                str(meas1) + ';' + str(meas2) 
                + str((
                    min(meas1.data.range()[0][0], meas2.data.range()[0][0]), 
                    max(meas1.data.range()[0][1], meas2.data.range()[0][1])
                ))
                + ".html"
            )
            
            create_directory(folder_path / 'HISTOGRAM')
            fig_hist, title_hist = density_plot([meas1, meas2])
            save_html_plotly(fig_hist, folder_path  / 'HISTOGRAM' / file_name)
            
            create_directory(folder_path / 'SCATTER')
            fig_scatter, title_scatter = scatter_plot([meas1, meas2])
            save_html_plotly(fig_scatter,  folder_path / 'SCATTER' / file_name)


    #Ploting trimmed pairs signals
    gruped_meas = group_meas(meas_list, ["meas_type", "meas_number"])

    for key, group in gruped_meas.items():
        filtered_meas = filter_meas(group)
        merged_meas = merge_grouped(filtered_meas)
        pairs = find_pairs(merged_meas)
        
        meas_type, meas_number = key
        
        selected_intervals = get_time_intervals(meas_number, meas_type)
        
        for meas1, meas2 in pairs:
            time_align_pair(meas1, meas2)
            
            # Iterate through the selected time intervals
            for (start_ms, end_ms), meas_state in selected_intervals.items():
                try:
                    trimmed_meas1 = trim_meas(meas1, start_ms, end_ms)
                    trimmed_meas2 = trim_meas(meas2, start_ms, end_ms)
                except ValueError as e:
                    print(f"Skipping pair due to invalid time range: {e}")
                    continue
            
                # Validate the trimmed Meas objects
                if not (validate_data(trimmed_meas1.data, 3) and validate_data(trimmed_meas2.data, 3)):
                    print(f"Validation failed for pair {trimmed_meas1} and {trimmed_meas2} in time interval {start_time}-{end_time}. Skipping this pair.")
                    continue
                
                folder_path = PLOTS_DIR / meas_type / str(meas_number) / "PAIRS_TRIMMED" / meas_state
                
                file_name = (
                    str(trimmed_meas1) + ';' + str(trimmed_meas2) 
                    + str((
                        min(trimmed_meas1.data.range()[0][0], trimmed_meas2.data.range()[0][0]), 
                        max(trimmed_meas1.data.range()[0][1], trimmed_meas2.data.range()[0][1])
                        ))
                    + ".html"
                )
                                
                create_directory(folder_path / 'SCATTER')
                fig_scatter, title_scatter = scatter_plot([trimmed_meas1, trimmed_meas2])
                save_html_plotly(fig_scatter,  folder_path / 'SCATTER' / file_name)
                
#%%
if __name__ == '__main__':
    main()