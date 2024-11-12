# Signal Data Processing and Correlation Analysis
___
## Project Overview
This project focuses on the analysis of signal data by calculating weighted correlations between pairs of measurements, with a special emphasis on heart rate signals. The methodology includes data loading, filtering, merging, and aligning of signals to assess the relationships between time-shifted measurements across various intervals.

## Key Features

### Data Loading
The project begins by importing and preprocessing signal data from designated directories. Measurements are organized by type and number, with each file name containing critical metadata:

`file name`: measurement number, relaxation or cooperation (r/w), gender (k/m), pair number, task registration number, and timestamp (start time).

### Measurement Filtering
Heart rate data can often contain artifacts, such as missed or erroneously registered beats due to sensor inaccuracies (e.g., Polar H10). To ensure the quality of the signal, two types of filtering are applied:
+ Removal of Outliers: Initially, RR intervals that fall outside the range of [−3×standard deviation,+3×standard deviation] are removed from the dataset. This step helps to eliminate extreme values that could distort the analysis.
+ Correction of Ectopic Beats: The second step involves identifying and removing ectopic heartbeats—those that occur prematurely in the cardiac cycle and are not triggered by the sinoatrial node but rather by spontaneous contractions of the heart muscle. A current RR interval RR[i] is considered ectopic and removed if it differs by more than 20% from the previous interval RR[i−1].

**Oryginal and Filtered Histograms**
![Sample pair histogram of RR signals](https://github.com/Hubert26/swps_synchronization_study/blob/main/images/rr_hist_image.png
"Sample pair histogram of RR signals")
![Sample pair histogram of NN signals](https://github.com/Hubert26/swps_synchronization_study/blob/main/images/nn_hist_image.png
"Sample pair histogram of NN signals")

### Signal Processing of NN Intervals into Different Metrics
+ **Instant Heart Rate (HR):** Calculated by converting NN intervals to beats per minute (BPM), providing an immediate measure of heart rate.
![Sample of interpolated HR signal durring first series of tasks](https://github.com/Hubert26/swps_synchronization_study/blob/main/images/hr_image.png
"Sample of interpolated HR signal durring first series of tasks")

+ **Standard Deviation of NN Intervals (SDNN):** Captures overall heart rate variability, indicating stability or variability in heart rhythm over the measurement period.
![Sample of interpolated SDNN signal durring first series of tasks](https://github.com/Hubert26/swps_synchronization_study/blob/main/images/sdnn_image.png
"Sample of interpolated SDNN signal durring first series of tasks")

+ **Root Mean Square of Successive Differences (RMSSD):** Calculated in 10-second windows with an 8-second overlap. This metric focuses on short-term variability and reflects parasympathetic nervous system activity.
![Sample of interpolated RMSSD signal durring first series of tasks](https://github.com/Hubert26/swps_synchronization_study/blob/main/images/rmssd_image.png
"Sample of interpolated RMSSD signal durring first series of tasks")

### Measurement Pairing
Signal data is paired according to specific criteria, such as measurement type and number. Each pair consists of data from both male and female participants.

### Measurement Shifting
In some pairs, one individual may lead the other in terms of signal patterns. To identify the strongest correlation, signals are time-shifted to account for this leading effect, optimizing alignment for accurate correlation analysis.

### Time Interval Processing
Measurements are segmented into specific time intervals, which correspond to different tasks performed by the participants. These intervals are customizable and are selected based on the measurement type and number.

### Measurement Merging
Occasionally, participants experience interruptions in data recording. When such disconnections occur, this project provides functionality to merge separate intervals of measurement, ensuring continuity in the data.

### Error Handling and Validation
Robust validation procedures are implemented to ensure that only valid and meaningful data are utilized for correlation analysis. Invalid inputs, such as incomplete time ranges or missing data, are logged, and the system automatically skips over problematic datasets.

### Results Handling
Correlation results are stored in pandas DataFrames, allowing for the identification of the best correlation for each time-shifted pair. The results, including the strongest correlations and their respective time shifts, are saved for further statistical analysis and interpretation.

## Code Structure

The project is organized into several directories and files, each serving a specific purpose to facilitate the synchronization study analysis. The main directory `src/` contains the source code and is subdivided into various modules:

- `utils/`: This directory includes utility modules such as `dataframe_utils.py`, `file_utils.py`, `math_utils.py`, `signal_utils.py`, `matplotlib_utils.py`, `ml_utils.py`, `plotly_utils.py`, and `string_utils.py`, providing essential functions for data manipulation, mathematical operations, plotting, and more.

- `classes.py`: This file defines the core classes used throughout the project, including `Data`, `Metadata`, and `Meas`, which encapsulate the data structures necessary for managing the time signals data.

- `main.py`: This is the entry point of the application where the main logic for data processing and analysis is executed.

- `functions.py`: This file contains various functions used for data processing and analysis tasks that support the main application logic.

- `config.py`: This file houses configuration settings and constants used across the project.

- `anova_data_preparation.py`: This module is dedicated to preparing data specifically for ANOVA analysis.

The `data/` directory contains subfolders for different types of measurement data, while the `results/` directory holds analysis results and generated plots. The `logs/` directory is used for logging messages during the execution of the code. Additionally, the root directory includes `environment.yml` for environment configuration, `.gitattributes` and `.gitignore` files for version control, and `README.md` for project documentation.

## Accessing Results

This project provides detailed signal data analysis and correlation results. You can explore and download the generated plots and results from the OneDrive link below:

[Download Plots and Results from OneDrive](https://1drv.ms/f/s!Aoqx7dFhq3qqw1Z6aStAwFQNVCWn?e=ge5kFP)

Please note that these files are available for download only, and no editing or deletion of files is permitted.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Hubert26/swps_synchronization_study/blob/main/LICENSE.txt) file for details.

