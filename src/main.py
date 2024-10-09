# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:45:31 2023

@author: Hubert Szewczyk
"""

import pandas as pd


from src.config import *
from src.config import intervals
from src.classes import Data, Metadata, Meas
import importlib
from src.functions import *
from src.utils.file_utils import list_file_paths, create_directory, delete_directory
#%%



#%%
if __name__ == '__main__':
    
    file_paths = list_file_paths(DATA_DIR)
    meas_list = load_data(file_paths)
    
    final_corr_results = pd.DataFrame(columns=['meas1', 'meas2', 'corr', 'shift_diff', 'meas_state'])
    final_interp_pairs = []
    
    gruped_meas = group_meas(meas_list, ["meas_type", "meas_number"])

    for key, group in gruped_meas.items():
        filtered_meas = filter_meas(group)
        merged_meas = merge_grouped(filtered_meas)
        pairs = find_pairs(merged_meas)
        
        meas_type, meas_number = key
        
        # Get time intervals for the given measurement number and type
        selected_intervals = get_time_intervals(meas_number, meas_type)
        
        for meas1, meas2 in pairs:
            
            best_corr_results = []
            best_interp_pairs = []
            
            # Iterate through the selected time intervals
            for (start_ms, end_ms), meas_state in selected_intervals.items():
                try:
                    # Determine the earlier starttime to use as reference
                    ref_starttime = min(meas1.metadata.starttime, meas2.metadata.starttime)
                    
                    trimmed_meas1 = trim_meas(meas1, start_ms, end_ms, starttime=ref_starttime)
                    trimmed_meas2 = trim_meas(meas2, start_ms, end_ms, starttime=ref_starttime)
                except ValueError as e:
                    logger.warning(f"Trimming meas to interval faild. Skipping pair due to invalid time range: {e}")
                    continue
            
                # Validate the trimmed Meas objects
                if not (validate_meas_data(trimmed_meas1, 3) and validate_meas_data(trimmed_meas2, 3)):
                    logger.warning(f"Validation input to calc corr failed for {trimmed_meas1} and {trimmed_meas2} with shift: 0.0 in time interval {start_ms}-{end_ms}. Skipping this pair.")
                    continue
                
                # Initialize lists to store correlation results
                corr_res_list = []
                interp_pair_list = []
                
                corr_res, interp_pair = calc_corr_weighted((meas1, meas2))
                if corr_res is not None and interp_pair is not None:
                    corr_res_list.append(pd.DataFrame([corr_res]))
                    interp_pair_list.append(interp_pair)
                
                # Iterate over various time shift values
                for shift_ms in range(1000, 5001, 1000):
                    shifted_meas1 = copy.deepcopy(meas1)
                    shifted_meas2 = copy.deepcopy(meas2)
                    shifted_meas1.shift_right(shift_ms)
                    shifted_meas2.shift_right(shift_ms)
                    
                    # Process shifted meas2
                    if not (validate_meas_data(trimmed_meas1, 3) and validate_meas_data(shifted_meas2, 3)):
                        logger.warning(f"Validation input to calc corr failed for {trimmed_meas1} and shifted {shifted_meas2} in time interval {start_ms}-{end_ms}. Skipping this pair.")
                        continue
                    else:
                        corr_res, interp_pair = calc_corr_weighted((trimmed_meas1, shifted_meas2))
                        if corr_res is not None and interp_pair is not None:
                            corr_res_list.append(pd.DataFrame([corr_res]))
                            interp_pair_list.append(interp_pair)
                        
                    # Process shifted meas1
                    if not (validate_meas_data(trimmed_meas2, 3) and validate_meas_data(shifted_meas1, 3)):
                        logger.warning(f"Validation input to calc corr failed for {trimmed_meas2} and shifted {shifted_meas1} in time interval {start_ms}-{end_ms}. Skipping this pair.")
                        continue
                    else:
                        corr_res, interp_pair = calc_corr_weighted((shifted_meas1, trimmed_meas2))
                        if corr_res is not None and interp_pair is not None:
                            corr_res_list.append(pd.DataFrame([corr_res]))
                            interp_pair_list.append(interp_pair)
                            
                # Concatenate the results for the current pair
                if corr_res_list and interp_pair_list:
                    corr_res_df = pd.concat(corr_res_list, ignore_index=False)
                    
                    # Find the best correlation result
                    max_abs_corr = corr_res_df['corr'].abs().max()
                    max_corr_rows = corr_res_df[corr_res_df['corr'].abs() == max_abs_corr]

                    if max_corr_rows.shape[0] > 1:
                        min_shift_index = max_corr_rows['shift_diff'].abs().idxmin()
                        best_corr_result = best_corr_result = max_corr_rows.iloc[min_shift_index]
                    else:
                        best_corr_result = max_corr_rows.iloc[0]
                    
                    # Find corresponding interp_pair for best_corr_result
                    for idx, row in corr_res_df.iterrows():
                        if (row['corr'] == best_corr_result['corr']) and (row['shift_diff'] == best_corr_result['shift_diff']):
                            best_interp_pair = interp_pair_list[idx]
                            break

                    # Save the best correlation result and corresponding interp_pair
                    best_corr_results.append(best_corr_result)
                    best_interp_pairs.append(best_interp_pair)
                    
            # Final concatenation of the best results
            if best_corr_results:
                best_corr_results = pd.DataFrame(best_corr_results)
                best_corr_results['meas_state'] = meas_state
            else:
                best_corr_results = pd.DataFrame(columns=['meas1', 'meas2', 'corr', 'shift_diff', 'meas_state'])

            final_corr_results = pd.concat([final_corr_results, best_corr_results], ignore_index=True)
            final_interp_pairs += best_interp_pairs

#%%
# =============================================================================
#     folder_path = PLOTS_DIR / 'INTERP_PAIRS'
#     
#     for meas1, meas2 in final_interp_pairs:
#         time_align_pair(meas1, meas2)
#         
#         file_name = (
#             str(meas1) + ';' + str(meas2) 
#             + str((
#                 min(meas1.data.range()[0][0], meas2.data.range()[0][0]), 
#                 max(meas1.data.range()[0][1], meas2.data.range()[0][1])
#             ))
#             + ".html"
#         )
#         
#         create_directory(folder_path / 'HISTOGRAM')
#         fig_hist, title_hist = density_plot([meas1, meas2])
#         save_html_plotly(fig_hist, folder_path  / 'HISTOGRAM' / file_name)
#         
#         create_directory(folder_path / 'SCATTER')
#         fig_scatter, title_scatter = scatter_plot([meas1, meas2])
#         save_html_plotly(fig_scatter,  folder_path / 'SCATTER' / file_name)
# =============================================================================

