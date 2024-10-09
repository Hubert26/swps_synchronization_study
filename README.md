# Signal Data Processing and Correlation Analysis
___
## Project Overview
This project focuses on the analysis of signal data by calculating weighted correlations between pairs of measurements, with a special emphasis on heart rate signals. The methodology includes data loading, filtering, merging, and aligning of signals to assess the relationships between time-shifted measurements across various intervals.

## Key Features
### Data Loading
The project begins by importing and preprocessing signal data from designated directories. Measurements are organized by type and number, with each file name containing critical metadata:

`file name`: measurement number, relaxation or cooperation (r/w), gender (k/m), pair number, task registration number, and timestamp (start time).

### Measurement Filtering
In some cases, heart rate data may contain artifacts, such as missed or erroneously registered beats due to sensor inaccuracies (e.g., Polar H10). To address this, two types of filtering are applied:
+ Removal of outliers beyond 2 standard deviations.
+ Correction of missed beats or spurious detections to ensure signal quality.

### Measurement Pairing
Signal data is paired according to specific criteria, such as measurement type and number. Each pair consists of data from both male and female participants.

### Measurement Merging
Occasionally, participants experience interruptions in data recording. When such disconnections occur, this project provides functionality to merge separate intervals of measurement, ensuring continuity in the data.

### Measurement Shifting
In some pairs, one individual may lead the other in terms of signal patterns. To identify the strongest correlation, signals are time-shifted to account for this leading effect, optimizing alignment for accurate correlation analysis.

### Time Interval Processing
Measurements are segmented into specific time intervals, which correspond to different tasks performed by the participants. These intervals are customizable and are selected based on the measurement type and number.

### Error Handling and Validation
Robust validation procedures are implemented to ensure that only valid and meaningful data are utilized for correlation analysis. Invalid inputs, such as incomplete time ranges or missing data, are logged, and the system automatically skips over problematic datasets.

### Results Handling
Correlation results are stored in pandas DataFrames, allowing for the identification of the best correlation for each time-shifted pair. The results, including the strongest correlations and their respective time shifts, are saved for further statistical analysis and interpretation.

## Code Structure
main.py: This file contains the main logic for the project, orchestrating the loading, processing, and correlation calculation. It manages the grouping, alignment, and analysis of measurements.
signal_visualization.py: Handles visualization tasks (if needed), including the plotting of signal data and comparison of measurement pairs over time.


