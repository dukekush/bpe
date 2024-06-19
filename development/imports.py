import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from textwrap import wrap
from sklearn.decomposition import PCA
import seaborn as sns 
import ruptures as rpt
from collections import Counter
from tslearn.metrics import dtw, gak

DATA_DIR = '../attack_pose_data/'
EXCEL_DATA_DIR = '../attack_pose_data/DATASETS/'
MAPPINGS = json.load(open(DATA_DIR+'translation_mappings.json'))


def aggregate_numeric_column(df, column_name, original_fps, target_fps):
    """
    Aggregates a single numeric column in a DataFrame based on video frame rates.
    
    Parameters:
    - df: DataFrame containing the video data.
    - column_name: The name of the numeric column to aggregate.
    - original_fps: The original frame rate of the video data.
    - target_fps: The target frame rate for aggregation.
    
    Returns:
    - A DataFrame with the aggregated column.
    - A dictionary mapping the original index to the new index.
    """
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    # Calculate the number of frames to aggregate to match the target frame rate
    factor = original_fps / target_fps
    if factor < 1:
        raise ValueError("Target FPS must be less than or equal to original FPS.")
    
    index_mapping = dict(zip(df.index, df.index // factor))

    # Aggregate the specified column
    aggregated_series = df[column_name].groupby(df.index // factor).mean()
    
    # Create a new DataFrame to return
    aggregated_df = pd.DataFrame(aggregated_series, columns=[column_name])

    return aggregated_df, index_mapping

def aggregate_all_numeric_columns(df, original_fps, target_fps):
    """
    Aggregates all numeric columns in a DataFrame based on video frame rates.
    
    Parameters:
    - df: DataFrame containing the video data.
    - original_fps: The original frame rate of the video data.
    - target_fps: The target frame rate for aggregation.
    
    Returns:
    - A DataFrame with the aggregated columns (returns only numeric columns).
    - A dictionary mapping the original index to the new index.
    """
    aggregated_dfs = []
    index_mappings = []
    
    for column_name in df.select_dtypes(include=[np.number]).columns:
        # if df[column_name].dtype == np.float64:
        aggregated_df, index_mapping = aggregate_numeric_column(df, column_name, original_fps, target_fps)
        aggregated_dfs.append(aggregated_df)
        index_mappings.append(index_mapping)
    
    return pd.concat(aggregated_dfs, axis=1), index_mappings


def bbox_normalize_joint_coordinates(df):

    # Extract all x and y columns
    x_columns = [col for col in df.columns if col.endswith('_x')]
    y_columns = [col for col in df.columns if col.endswith('_y')]

    # Compute the max and min values for x and y across all joints
    max_x = df[x_columns].values.max()
    min_x = df[x_columns].values.min()
    max_y = df[y_columns].values.max()
    min_y = df[y_columns].values.min()

    # Define a function to normalize values
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)
    
    # Normalize all x and y coordinates
    for col in x_columns:
        df.loc[:, col + '_normalized'] = df[col].apply(normalize, args=(min_x, max_x))
    
    for col in y_columns:
        df.loc[:, col + '_normalized'] = df[col].apply(normalize, args=(min_y, max_y))

    return df


def aggregate_phases(df, original_fps, target_fps):
    def aggfunc(group):
        if 'attack' in group.values:
            return 'attack'
        else:
            # Compute the mode with pandas.Series.mode, which handles non-numeric data
            mode_values = group.mode()
            return mode_values.iloc[0] if not mode_values.empty else None
        
    factor = original_fps / target_fps
    if factor < 1:
        raise ValueError("Target FPS must be less than or equal to original FPS.")
    
    df['index'] = df.index // factor
        
    return df.groupby('index')['phase'].agg(aggfunc)


def plot_sequence(df, frame_number, normalised=False, ax=None, title='Skeleton Connections', xlabel='X', ylabel='Y', **kwargs):
    connections = [
        ('nose', 'mid_shoulder'),
        # Spine
        ("mid_hip", "mid_shoulder"),
        # Right Arm
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        # Left Arm
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        # Right Leg
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        # Left Leg
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle")
    ]

    # Plotting
    if ax is None:
        fig, ax = plt.subplots()

    for start, end in connections:
        if normalised:
            start_x = df[f'{start}_x_normalized'].values[frame_number]
            start_y = df[f'{start}_y_normalized'].values[frame_number]
            end_x = df[f'{end}_x_normalized'].values[frame_number]
            end_y = df[f'{end}_y_normalized'].values[frame_number]
        else:
            start_x = df[f'{start}_x'].values[frame_number]
            start_y = df[f'{start}_y'].values[frame_number]
            end_x = df[f'{end}_x'].values[frame_number]
            end_y = df[f'{end}_y'].values[frame_number]
        
        out = ax.plot([start_x, end_x], [start_y, end_y], 'o-', **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=16)
    ax.invert_yaxis()
    
    return out 

def normalise_dataframe_by_spine(df):
    # Calculate midpoints
    df['mid_hip_x'] = (df['left_hip_x'] + df['right_hip_x']) / 2
    df['mid_hip_y'] = (df['left_hip_y'] + df['right_hip_y']) / 2
    df['mid_shoulder_x'] = (df['left_shoulder_x'] + df['right_shoulder_x']) / 2
    df['mid_shoulder_y'] = (df['left_shoulder_y'] + df['right_shoulder_y']) / 2

    # Calculate spine length
    df['spine_length'] = (
        (df['mid_shoulder_x'] - df['mid_hip_x'])**2 +\
        (df['mid_shoulder_y'] - df['mid_hip_y'])**2
        )**0.5

    # Normalize the data
    for joint in ['nose','right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist', 'left_elbow', 'left_shoulder', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'mid_hip', 'mid_shoulder']:
        df[f'{joint}_x_normalized'] = (df[f'{joint}_x'] - df['mid_hip_x']) / df['spine_length']
        df[f'{joint}_y_normalized'] = (df[f'{joint}_y'] - df['mid_hip_y']) / df['spine_length']

    # Selecting columns for demonstration
    normalized_columns = [col for col in df.columns if 'normalized' in col]
    df[normalized_columns].head()

    return df

def plot_frame(df, frame_number):
    connections = [
        # Nose
        ('nose', 'mid_shoulder'),
        # Spine
        ("mid_hip", "mid_shoulder"),
        # Right Arm
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        # Left Arm
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        # Right Leg
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        # Left Leg
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle")
    ]

    # Plotting
    fig, ax = plt.subplots(1, 2)
    for start, end in connections:
        # Extracting the starting and ending points for each connection from the DataFrame
        start_x = df[f'{start}_x'].values[frame_number]
        start_y = df[f'{start}_y'].values[frame_number]
        end_x = df[f'{end}_x'].values[frame_number]
        end_y = df[f'{end}_y'].values[frame_number]
        
        ax[0].plot([start_x, end_x], [start_y, end_y], 'ro-')  # 'ro-' means red color, circle markers, and solid line style
        
        start_x = df[f'{start}_x_normalized'].values[frame_number]
        start_y = df[f'{start}_y_normalized'].values[frame_number]
        end_x = df[f'{end}_x_normalized'].values[frame_number]
        end_y = df[f'{end}_y_normalized'].values[frame_number]
        
        ax[1].plot([start_x, end_x], [start_y, end_y], 'ro-')

    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].set_title('Skeleton Connections')
    ax[0].invert_yaxis()
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[1].set_title('Skeleton Connections (Normalized)')
    ax[1].invert_yaxis()
    plt.show()