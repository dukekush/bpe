import os 
import math
import numpy as np
import pandas as pd
import argparse
import os
import math
import imageio as io
import json
import matplotlib.pyplot as plt
import ruptures as rpt
from tslearn.metrics import dtw_path, dtw
from sklearn.metrics.pairwise import cosine_similarity
import cv2

from config import *

# os.chdir(PROJECT_GLOBAL_PATH)

from bpe import Config
from bpe.functional.motion import preprocess_motion2d_rc, annotations2motion
from bpe.similarity_analyzer import SimilarityAnalyzer
from bpe.functional.visualization import preprocess_sequence

phase_names = {
    0: 'Preparation + Motion Start',
    1: 'Final Steps + Jump Start',
    2: 'Jump Start + Ball Contact',
    3: 'Landing',
}

def config_parser():
    video1 = ''
    video2 = ''
    img1_height = 1080
    img1_width = 1920
    img2_height = 1080
    img2_width = 1920

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="", required=True, help="path to dataset dir")
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")
    parser.add_argument('--video1', type=str, required=True, help="video1 mp4 path", default=None)
    parser.add_argument('--video2', type=str, required=True, help="video2 mp4 path", default=None)
    parser.add_argument('-h1', '--img1_height', type=int, help="video1's height", default=480)
    parser.add_argument('-w1', '--img1_width', type=int, help="video1's width", default=854)
    parser.add_argument('-h2', '--img2_height', type=int, help="video2's height", default=480)
    parser.add_argument('-w2', '--img2_width', type=int, help="video2's width", default=854)
    parser.add_argument('-pad2', '--pad2', type=int, help="video2's start frame padding", default=0)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument('--out_path', type=str, default='./visual_results', required=False)
    parser.add_argument('--out_filename', type=str, default='twice.mp4', required=False)
    parser.add_argument('--use_flipped_motion', action='store_true',
                        help="whether to use one decoder per one body part")
    parser.add_argument('--use_invisibility_aug', action='store_true',
                        help="change random joints' visibility to invisible during training")
    parser.add_argument('--debug', action='store_true', help="limit to 500 frames")
    # related to video processing
    parser.add_argument('--video_sampling_window_size', type=int, default=16,
                        help='window size to use for similarity prediction')
    parser.add_argument('--video_sampling_stride', type=int, default=16,
                        help='stride determining when to start next window of frames')
    parser.add_argument('--use_all_joints_on_each_bp', action='store_true',
                        help="using all joints on each body part as input, as opposed to particular body part")

    parser.add_argument('--similarity_measurement_window_size', type=int, default=1,
                        help='measuring similarity over # of oversampled video sequences')
    parser.add_argument('--similarity_distance_metric', choices=["cosine", "l2"], default="cosine")
    parser.add_argument('--privacy_on', action='store_true',
                        help='when on, no original video or sound in present in the output video')
    parser.add_argument('--thresh', type=float, default=0.5, help='threshold to seprate positive and negative classes')
    parser.add_argument('--connected_joints', action='store_false', help='connect joints with lines in the output video')


    args = parser.parse_args([
        '--data_dir', DATA_PATH, 
        '--model_path', MODEL_PATH,
        '--video1', video1,
        '--video2', video2,
        '--img1_height', str(img1_height),
        '--img1_width', str(img1_width),
        '--img2_height', str(img2_height),
        '--img2_width', str(img2_width),
        ])
    
    return args



def pose_df_to_dict(pose_df):

	annot = {'annotations': []}

	for i, row in pose_df.iterrows():
		keypoints = [
		row['nose_x'],
		row['nose_y'],
		row['nose_p'],
		
		row['left_eye_x'],
		row['left_eye_y'],
		row['left_eye_p'],
		
		row['right_eye_x'],
		row['right_eye_y'],
		row['right_eye_p'],
		
		row['left_ear_x'],
		row['left_ear_y'],
		row['left_ear_p'],
		
		row['right_ear_x'],
		row['right_ear_y'],
		row['right_ear_p'],
		
		row['left_shoulder_x'],
		row['left_shoulder_y'],
		row['left_shoulder_p'],
		
		row['right_shoulder_x'],
		row['right_shoulder_y'],
		row['right_shoulder_p'],
		
		row['left_elbow_x'],
		row['left_elbow_y'],
		row['left_elbow_p'],
		
		row['right_elbow_x'],
		row['right_elbow_y'],
		row['right_elbow_p'],
		
		row['left_wrist_x'],
		row['left_wrist_y'],
		row['left_wrist_p'],
		
		row['right_wrist_x'],
		row['right_wrist_y'],
		row['right_wrist_p'],
		
		row['left_hip_x'],
		row['left_hip_y'],
		row['left_hip_p'],
		
		row['right_hip_x'],
		row['right_hip_y'],
		row['right_hip_p'],
		
		row['left_knee_x'],
		row['left_knee_y'],
		row['left_knee_p'],
		
		row['right_knee_x'],
		row['right_knee_y'],
		row['right_knee_p'],
		
		row['left_ankle_x'],
		row['left_ankle_y'],
		row['left_ankle_p'],
		
		row['right_ankle_x'],
		row['right_ankle_y'],
		row['right_ankle_p'],
	]
		frame_num = row['frame_number']

		category_id = 1
		bbox = [0 for _ in range(4)]
		score = 1
		area = 0
		b_score = 1
		object_id = 0
		objects = [
			{
				'category_id': category_id,
				'bbox': bbox,
				'score': score,
				'keypoints': keypoints,
				'area': area,
				'b_score': b_score,
				'object_id': object_id,
			}
		]
		annot['annotations'].append(
			{
				'frame_num': frame_num,
				'objects': objects,
			}
		) 
	return annot


class PhaseComparison:
    '''
    A class to compare two videos based on their pose data and generate a video output with comparison results overlayed.
    '''

    JOINTS = [
    'right_shoulder', 'right_elbow', 'right_wrist', 
    'nose', 
    'left_shoulder', 'left_elbow', 'left_wrist', 
    'right_hip', 'right_knee', 'right_ankle', 
    'left_hip', 'left_knee', 'left_ankle'
    ]

    def __init__(self):
        # Load setup files
        self.mappings = json.load(open('attack_pose_data/translation_mappings.json'))
        self.args = config_parser()
        self.config = Config(self.args)
        self.similarity_analyzer = SimilarityAnalyzer(self.config, MODEL_PATH)
        self.mean_pose_bpe = np.load(os.path.join(self.args.data_dir, 'meanpose_rc_with_view_unit64.npy'))
        self.std_pose_bpe = np.load(os.path.join(self.args.data_dir, 'stdpose_rc_with_view_unit64.npy'))


    def generate_video_comparison_output(self, 
                                         output_video_path,
                                         reference_excel_path,
                                         comparison_excel_path,
                                         comparison_video_path,
                                         num_phases=4,
                                         slow_factor=1,
                                         min_score=0.43):
        '''
        Function to generate video comparison output.

        Parameters:
        ----------
        output_video_path : str
            The path to save the output video.
        reference_excel_path : str
            The path to the reference video's pose excel data.
        comparison_excel_path : str
            The path to the comparison video's pose excel data.
        comparison_video_path : str
            The path to the comparison video file.
        num_phases : int
            The number of phases to compare.
        slow_factor : int
            The slow factor for the output video (output video will have original fps divided by this). The larger the number, the slower the output video.
        min_score : float
            The minimum similarity score to consider. Used for color mapping in the output video. Min score will be mapped to red, max score (1) will be mapped to green.

        Returns:
        --------
        None
        '''
        
        self._compare_videos(reference_excel_path, comparison_excel_path, num_phases)
        self._generate_video(output_video_path, comparison_video_path, comparison_excel_path, slow_factor, min_score)


    def get_comparison_scores(self, reference_excel_path, comparison_excel_path, num_phases):
        '''
        Function to get the comparison scores between two videos. Input is the path to the pose excel data for both videos.
        
        Parameters:
        ----------
        reference_excel_path : str
            The path to the reference video's pose excel data.
        comparison_excel_path : str
            The path to the comparison video's pose excel data.
        num_phases : int
            The number of phases to segment videos into.
        
        Returns:
        --------
        dict
            A dictionary containing the comparison scores for each body part in each phase.
        '''
        self._compare_videos(reference_excel_path, comparison_excel_path, num_phases)
        return self.comparison_data[1]
    

    def _compare_videos(self, reference_excel_path, comparison_excel_path, num_phases):
        self.comparison_data = compare_videos(
            reference_excel_path, 
            comparison_excel_path, 
            num_phases, 
            self.JOINTS, 
            self.config, 
            self.args, 
            self.similarity_analyzer, 
            self.mean_pose_bpe, 
            self.std_pose_bpe
            )
        
        
    def _generate_video(self, output_video_path, comparison_video_path, comparison_excel_path, slow_factor=1, min_score=0):
        video = io.get_reader(comparison_video_path)
        video_name = comparison_video_path.split('/')[-1]
        fps = video.get_meta_data()['fps']
        size = video.get_meta_data()['size']

        video = iter(video)

        frame_idx = 0
        sequence, _ = load_pose_data(comparison_excel_path, self.mappings)
        sequence = preprocess_df_for_bpe(sequence, self.config)
        bkpts = self.comparison_data[-1]

        phase = 0
        all_frames = []
        for frame_idx in range(sequence.shape[2]):

            frame = sequence[:, :, frame_idx]
            canvas = next(video)

            phase_sims = self.comparison_data[1][f'phase_{phase}']
            colors_per_joint = get_colors_per_joint(phase_sims, min_score) 
            
            draw_seq(canvas, frame, colors_per_joint, size)

            put_similarity_score_in_video(canvas, phase_sims, size, min_score)
            put_filename_in_video(canvas, video_name, size)
            put_phase_in_video(canvas, phase, size, phase_names)

            if frame_idx in bkpts:
                phase += 1
            all_frames.append(canvas)

        io.mimwrite(output_video_path, all_frames, fps=fps/slow_factor)


# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Preprocessing functions ----------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

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


def load_pose_data(pose_df_path, mappings):
    df = pd.read_excel(pose_df_path, index_col=0)
    df.rename(columns=mappings, inplace=True)
    fps = df['fps'].values[0]
    return df, fps


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


def get_normalised_kepoints(aggregated_pose_df, joints):
    norm_df = bbox_normalize_joint_coordinates(aggregated_pose_df[[joint + '_x' for joint in joints] + [joint + '_y' for joint in joints]].copy())
    normalised_keypoints_y = np.array([norm_df[joint + '_y_normalized'].values for joint in joints])
    normalised_keypoints_x = np.array([norm_df[joint + '_x_normalized'].values for joint in joints])
    normalised_keypoints_all = np.concatenate([normalised_keypoints_x, normalised_keypoints_y], axis=0)
    return normalised_keypoints_all.T, normalised_keypoints_x.T, normalised_keypoints_y.T


def preprocess_df_for_bpe(aggregated_pose_df, config):
    annot = pose_df_to_dict(aggregated_pose_df)
    seq = annotations2motion(config.unique_nr_joints, annot['annotations'], scale=1)
    seq = preprocess_sequence(seq)
    return seq


# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Breakpoints functions ------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

def get_breakpoints(normalised_keypoints, num_phases):
    bkpts_algorithm = rpt.KernelCPD(kernel="rbf", min_size=14)
    bkpts = bkpts_algorithm.fit_predict(normalised_keypoints, n_bkps=num_phases - 1)
    return bkpts


def reindex(bkpts, index):
    result_reindexed = []
    for bkp in bkpts:
        for k,v in index.items():
            if bkp == v:
                result_reindexed.append(k)
                break
    return result_reindexed


# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Comparison functions -------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

def get_embeddings(seq, window_size, stride, similarity_analyzer, mean_pose_bpe, std_pose_bpe, config, args):
    seq = preprocess_motion2d_rc(seq, mean_pose_bpe, std_pose_bpe, use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)
    seq = seq.to(config.device)
    seq_features = similarity_analyzer.get_embeddings(seq, video_window_size=window_size, video_stride=stride)
    return seq_features


def compute_dtw_cosine_similarity(sequence1, sequence2):
    """
    Computes the cosine similarity between two sequences using Dynamic Time Warping (DTW).

    Parameters:
    - sequence1: The first sequence.
    - sequence2: The second sequence.

    Returns:
    - The cosine similarity between the two sequences.
    """
    # Compute DTW path
    path, _ = dtw_path(sequence1.reshape(-1, 1), sequence2.reshape(-1, 1))

    # Extract aligned sequences
    aligned_sequence1 = [sequence1[idx] for idx, _ in path]
    aligned_sequence2 = [sequence2[idx] for _, idx in path]

    # Convert aligned sequences to NumPy arrays
    aligned_sequence1 = np.array(aligned_sequence1)
    aligned_sequence2 = np.array(aligned_sequence2)

    # Reshape sequences for cosine similarity calculation
    aligned_sequence1 = aligned_sequence1.reshape(1, -1)
    aligned_sequence2 = aligned_sequence2.reshape(1, -1)

    # Compute cosine similarity
    cos_sim = cosine_similarity(aligned_sequence1, aligned_sequence2)[0][0]

    return cos_sim


def compute_dtw_cosine_similarity_per_phase(seq_ref, seq_comp, breakpoints_ref, breakpoints_comp, config, args, similarity_analyzer, mean_pose_bpe, std_pose_bpe):
    start_ref = 0
    start_comp = 0
    sims_dtw = {}
    sims_dtw_cos = {}
    for i, (end_ref, end_comp) in enumerate(zip(breakpoints_ref, breakpoints_comp)):
        length_ref = end_ref - start_ref
        length_comp = end_comp - start_comp
        seq_ref_features = get_embeddings(seq_ref[:, :, start_ref:end_ref], length_ref, length_ref, similarity_analyzer, mean_pose_bpe, std_pose_bpe, config, args)
        seq_comp_features = get_embeddings(seq_comp[:, :, start_comp:end_comp], length_comp, length_comp, similarity_analyzer, mean_pose_bpe, std_pose_bpe, config, args)
        start_ref = end_ref 
        start_comp = end_comp

        sims_dtw[f'phase_{i}'] = {}
        sims_dtw_cos[f'phase_{i}'] = {}
        for bp_i, bp in enumerate(config.body_part_names):
            sims_dtw[f'phase_{i}'][bp] = dtw(seq_ref_features[0][bp_i], seq_comp_features[0][bp_i])
            sims_dtw_cos[f'phase_{i}'][bp] = compute_dtw_cosine_similarity(seq_ref_features[0][bp_i], seq_comp_features[0][bp_i])

    return sims_dtw, sims_dtw_cos


def compare_videos(pose_reference_df_path, pose_comparison_df_path, NUM_PHASES, joints, config, args, similarity_analyzer, mean_pose_bpe, std_pose_bpe):
    mappings = json.load(open('attack_pose_data/translation_mappings.json'))

    pose_reference_df, reference_fps = load_pose_data(pose_reference_df_path, mappings)
    pose_comparison_df, comparison_fps = load_pose_data(pose_comparison_df_path, mappings)

    aggregated_pose_reference_df, _ = aggregate_all_numeric_columns(pose_reference_df, reference_fps, reference_fps)
    aggregated_pose_comparison_df, index = aggregate_all_numeric_columns(pose_comparison_df, comparison_fps, comparison_fps)

    normalised_keypoints_ref, _, _ = get_normalised_kepoints(aggregated_pose_reference_df, joints)
    normalised_keypoints_comp, _, _ = get_normalised_kepoints(aggregated_pose_comparison_df, joints)

    breakpoints_ref = get_breakpoints(normalised_keypoints_ref, NUM_PHASES)
    breakpoints_comp = get_breakpoints(normalised_keypoints_comp, NUM_PHASES)

    breakpoints_comp_reindexed = reindex(breakpoints_comp, index[0])

    seq_ref = preprocess_df_for_bpe(aggregated_pose_reference_df, config)
    seq_comp = preprocess_df_for_bpe(aggregated_pose_comparison_df, config)

    sims_dtw, sims_dtw_cos = compute_dtw_cosine_similarity_per_phase(seq_ref, seq_comp, breakpoints_ref, breakpoints_comp, config, args, similarity_analyzer, mean_pose_bpe, std_pose_bpe)

    return sims_dtw, sims_dtw_cos, breakpoints_ref, breakpoints_comp_reindexed


# ------------------------------------------------------------------------------------------------------------------------------
# --------------------------- Visualisation functions --------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

def sample_color(score, min_score=0.5, max_score=1.0):
    """
    Sample a color from a red-to-yellow-to-green palette based on a similarity score.
    
    Args:
        score (float): The similarity score to map to a color. Should be between min_score and max_score.
        min_score (float): The minimum similarity score, mapped to full red.
        max_score (float): The maximum similarity score, mapped to full green.

    Returns:
        tuple: A tuple representing the RGB color.
    """
    # Ensure score is within bounds
    if score < min_score:
        score = min_score
    elif score > max_score:
        score = max_score
    
    # Calculate the normalized score
    normalized_score = (score - min_score) / (max_score - min_score)
    
    if normalized_score <= 0.5:
        # Interpolate from red to yellow
        factor = normalized_score / 0.5
        red = 255
        green = int(factor * 255)
        blue = 0
    else:
        # Interpolate from yellow to green
        factor = (normalized_score - 0.5) / 0.5
        red = int((1 - factor) * 255)
        green = 255
        blue = 0
    
    return (red, green, blue)

def visualize_palette(min_score=0.5, max_score=1.0, num_samples=100):
    """
    Visualize the palette from red to yellow to green.
    
    Args:
        min_score (float): The minimum similarity score.
        max_score (float): The maximum similarity score.
        num_samples (int): The number of samples to generate in the palette.
    """
    scores = np.linspace(min_score, max_score, num_samples)
    colors = [sample_color(score, min_score, max_score) for score in scores]
    
    # Normalize colors to [0, 1] for displaying with matplotlib
    colors = [(r/255, g/255, b/255) for r, g, b in colors]
    
    # Create an image to display the palette
    img = np.array([colors])
    
    plt.figure(figsize=(10, 2))
    plt.imshow(img, aspect='auto')
    plt.title(f'Color Palette from Red to Yellow to Green (Scores {min_score} to {max_score})')
    plt.axis('off')
    plt.show()


def get_colors_per_joint(similarity_per_bp, min_score):
    color_per_joint = np.tile([0, 255, 0], (15, 1))

    '''
    joints order : 
    [nose, neck,
    right_shoulder, right_elbow, right_wrist,
    left_shoulder, left_elbow, left_wrist,
    mid_hip,
    right_hip, right_knee, right_ankle,
    left_hip, left_knee, left_ankle]
    '''
    for _, bp in enumerate(similarity_per_bp.keys()):
        similarity = round(similarity_per_bp[bp], 2)
        cur_joint_color_left_side = sample_color(similarity, min_score, 1)
        cur_joint_color_right_side_torso = sample_color(similarity, min_score, 1)

        if bp == 'torso':
            color_per_joint[[0, 1, 8]] = cur_joint_color_right_side_torso
        elif bp == 'ra':
            color_per_joint[[2, 3, 4]] = cur_joint_color_right_side_torso
        elif bp == 'la':
            color_per_joint[[5, 6, 7]] = cur_joint_color_left_side
        elif bp == 'rl':
            color_per_joint[[9, 10, 11]] = cur_joint_color_right_side_torso
        elif bp == 'll':
            color_per_joint[[12, 13, 14]] = cur_joint_color_left_side
        else:
            raise KeyError('Wrong body part key')
    return color_per_joint


def put_similarity_score_in_video(img, similarity_per_bp, video_size, min_score):
    width, height = video_size
    font_scale = height / 720
    thickness = max(1, int(height / 360))
    for bp_idx, bp in enumerate(similarity_per_bp.keys()):
        similarity = round(similarity_per_bp[bp], 2)
        color = sample_color(similarity, min_score, 1)
        y_coord, x_coord = int(0.05 * height + 0.05 * height * bp_idx), int(0.02 * width)
        cv2.putText(img, '{}:{:.2f}'.format(bp, similarity), (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, thickness)


def put_filename_in_video(img, filename, video_size):
    width, height = video_size
    font_scale = height / 720
    thickness = max(1, int(height / 360))
    x_coord, y_coord = int(0.15 * width), int(0.05 * height)
    cv2.putText(img, filename, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


def put_phase_in_video(img, phase, video_size, phase_names):
    width, height = video_size
    font_scale = height / 720  # Scale font size based on height (assuming 720p as baseline)
    thickness = max(1, int(height / 360))  # Adjust thickness based on height
    base_y_coord = int(0.05 * height)  # Starting y-coordinate below the filename
    x_coord = int(0.65 * width)  # Set x-coordinate to place text in the top right corner

    for idx, (phase_id, phase_name) in enumerate(phase_names.items()):
        color = (0, 255, 0) if phase_id == phase else (255, 0, 0)  # Green for current phase, red for others
        y_coord = base_y_coord + int(0.05 * height) * idx
        cv2.putText(img, phase_name, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_seq(img, frame_seq, color_per_joint, video_size, left_padding=0):
    assert len(frame_seq) == len(color_per_joint)
    width, height = video_size
    thickness = max(1, int(height / 360))
    draw_connected_joints(img, frame_seq, color_per_joint, left_padding, thickness)

    # add joints visualization
    stickwidth = thickness
    for joint_idx, joint_xy in enumerate(frame_seq):
        x_coord, y_coord = joint_xy

        color = [int(i) for i in color_per_joint[joint_idx]]
        cv2.circle(img, (left_padding + int(x_coord), int(y_coord)), stickwidth, color, 3)


def draw_connected_joints(canvas, joints, colors, left_padding, stickwidth=5):
    # connect joints with lines
    # ([nose, neck,
    # right_shoulder, right_elbow, right_wrist,
    # left_shoulder, left_elbow, left_wrist,
    # mid_hip,
    # right_hip, right_knee, right_ankle,
    # left_hip, left_knee, left_ankle,])
    limb_seq = [[0, 1], [1, 8], [5, 6], [6, 7], [2, 3], [3, 4], [12, 13], [13, 14], [9, 10], [10, 11]]

    for i in range(len(limb_seq)):
        X = (int(joints[limb_seq[i][0]][0] + left_padding), int(joints[limb_seq[i][1]][0] + left_padding))
        Y = (int(joints[limb_seq[i][0]][1]), int(joints[limb_seq[i][1]][1]))

        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
        polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        color = [int(i) for i in colors[limb_seq[i][0]]]
        cv2.fillConvexPoly(canvas, polygon, color)

    return canvas