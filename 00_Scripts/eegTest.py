# %% load packages
import numpy as np
import mne
import os
import glob
import pandas as pd
from scipy.spatial import distance

#%% Define excluded subject numbers
excluded_subjects = ['02', '03']  # Define your list of excluded subject numbers

#%% Load in data files
# Directory path to search recursively
# directory = 'F:\\DR-RisSen-05\\EEG'

#RDM DIREC
directory = '/QRISdata/Q4364/DR-RisSen-05/00_SourceData/EEG/'

# List to store file paths
file_paths = glob.glob(os.path.join(directory, '**', '*MAIN_eeg*'), recursive=True)

# Filter file paths based on excluded subjects
file_paths_filtered = []
for path in file_paths:
    # Extract subject number from file path
    subject_number = os.path.basename(path).split('_')[0].split('-')[1]
    
    # Check if subject number is in excluded list
    if subject_number not in excluded_subjects:
        file_paths_filtered.append(path)
    else:
        print(f"Excluding subject {subject_number} from analysis.")

# Print or use the list of filtered file paths
print("Found files:")
for path in file_paths_filtered:
    print(path)

# Load all raw files from filtered paths
ind_subs = [mne.io.read_raw(file, preload=False) for file in file_paths_filtered]

#%% clip data files so that their lengths are equal

# Function to load raw files and prepare them
def load_and_prepare_raw_files(raw_files):
    raw_data_list = []
    min_length = float('inf')
    
    # Load each raw file and determine the minimum length
    for raw in raw_files:
        data = raw.get_data()
        raw_data_list.append(data)
        if data.shape[1] < min_length:
            min_length = data.shape[1]
    
    # Truncate each data array to the minimum length
    truncated_data_list = [data[:, :min_length] for data in raw_data_list]
    
    return truncated_data_list, min_length, raw_files[0].info

# Load and prepare the raw files
truncated_data_list, min_length, info = load_and_prepare_raw_files(ind_subs)
print("Minimum length of data:", min_length)

#%% Combine channels within each participant

# List to store combined signals for each participant
combined_signals = []

# Combine channels for each participant, excluding specific channels
channels_to_exclude = info['ch_names'][-9:]  # Channels to exclude

for data, raw in zip(truncated_data_list, ind_subs):
    # Get indices of channels to exclude
    exclude_indices = [raw.ch_names.index(ch) for ch in channels_to_exclude]
    include_indices = [i for i in range(data.shape[0]) if i not in exclude_indices]
    
    # Average only the included channels
    combined_signal = np.mean(data[include_indices, :], axis=0)  # Average across included channels
    combined_signals.append(combined_signal)

# Stack combined signals along a new axis: time x participants
combined_data_3d = np.stack(combined_signals, axis=-1)

#%% Compute the average across participants

# Compute the average across participants (axis=-1)
average_participant = np.mean(combined_data_3d, axis=-1)

# Create RawArray with a single channel
info = mne.create_info(ch_names=['average_signal'], sfreq=raw.info['sfreq'])
raw_average_participant = mne.io.RawArray(average_participant[np.newaxis, :], info)

# Save the average participant signal
deriv_save_path = '/home/s4577680/Desktop/Scratch/s4577680/DR-RisSen-05/Outputs/'
average_participant_save_path = os.path.join(os.path.dirname(deriv_save_path), 'avgEEGSignal_EEG.bdf')
raw_average_participant.save(average_participant_save_path)
print(f"Saved average participant signal to: {average_participant_save_path}")

# Print the info of the average participant
print(raw_average_participant.info)

#%% Initialize a list to store distances and subject numbers
distance_data = []

# Calculate distances for each participant
for raw_participant in ind_subs:
    # Get all channel names except those to exclude
    all_channels = [ch for ch in raw_participant.ch_names if ch not in channels_to_exclude]
    groups = {'average_signal': [raw_participant.ch_names.index(ch) for ch in all_channels]}
    
    raw_participant_signal = mne.channels.combine_channels(raw_participant, groups=groups)
    
    # Ensure signals have the same length
    min_length = min(raw_average_participant.get_data().shape[1], raw_participant_signal.get_data().shape[1])
    average_signal_data = raw_average_participant.get_data()[:, :min_length]
    participant_signal_data = raw_participant_signal.get_data()[:, :min_length]
    
    # Flatten the signals to convert them to 1D arrays
    average_signal_flat = average_signal_data.flatten()
    participant_signal_flat = participant_signal_data.flatten()
    
    # Compute the Euclidean distance
    euclidean_dist = distance.euclidean(average_signal_flat, participant_signal_flat)
    
    # Get subject number
    subj_num = raw_participant.filenames[0].split('_')[2].split('-')[3]

    # Store the distances and subject number
    distance_data.append({'subject_number': subj_num, 'euclidean_distance': euclidean_dist})

# Create a DataFrame
distance_df = pd.DataFrame(distance_data)

# Calculate mean and standard deviation of distances
mean_distance = distance_df['euclidean_distance'].mean()
std_distance = distance_df['euclidean_distance'].std()

# Add column for distance within 3 standard deviations
distance_df['outside_3std'] = distance_df.apply(lambda row: np.abs(row['euclidean_distance'] - mean_distance) > 3 * std_distance, axis=1)

# Save the distance DataFrame to the same directory as the average participant signal
distance_df_save_path = os.path.join(os.path.dirname(deriv_save_path), 'distance_dataframe.csv')
distance_df.to_csv(distance_df_save_path, index=False)
print(f"Saved distance DataFrame to: {distance_df_save_path}")


# Print the DataFrame
print(distance_df)
# %%
