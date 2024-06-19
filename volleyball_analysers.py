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

from config import *

os.chdir(PROJECT_GLOBAL_PATH)

from bpe import Config
from bpe.functional.motion import preprocess_motion2d_rc, annotations2motion
from bpe.similarity_analyzer import SimilarityAnalyzer
from bpe.functional.visualization import get_colors_per_joint, put_similarity_score_in_video, preprocess_sequence, draw_seq, put_filename_in_video


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


class VolleyballAnalyser:
    def __init__(self, vid1_pose_file, vid2_pose_file, window_size=30, stride_size=30, translation_mappings_path='attack_pose_data/translation_mappings.json'):
        '''
        Constructs all the necessary attributes for the VolleyballAnalyser object.
        
        Parameters:
        -----------
        vid1_pose_file: str
            Path to the first video's pose excel file.
        vid2_pose_file: str
            Path to the second video's pose excel file.
        window_size: int, default=30
            Window size used for similarity prediction.
        stride: int, default=30
            Stride size used for similarity prediction.
        translation_mappings_path: str, default='attack_pose_data/translation_mappings.json'
            Path to the translation mappings file.
        '''
        self.args = config_parser()
        self.config = Config(self.args)
        self.similarity_analyzer = SimilarityAnalyzer(self.config, self.args.model_path)

        assert window_size >= 14, 'Window size must be greater than 14'
        assert stride_size >= 1, 'Stride must be greater than 0'
        self.args.video_sampling_window_size = window_size
        self.args.video_sampling_stride = stride_size
        
        self.vid1_pose_file = vid1_pose_file.split('/')[-1]
        self.vid2_pose_file = vid2_pose_file.split('/')[-1]
        self.vid1 = pd.read_excel(vid1_pose_file, index_col=0)
        self.vid2 = pd.read_excel(vid2_pose_file, index_col=0)
        self._adjust_vid_length()

        self.translation_mappings = json.load(open(translation_mappings_path))

        self.seq1 = self._load_seq_from_df(self.vid1)
        self.seq2 = self._load_seq_from_df(self.vid2)

        self.mean_pose_bpe = np.load(os.path.join(self.args.data_dir, 'meanpose_rc_with_view_unit64.npy'))
        self.std_pose_bpe = np.load(os.path.join(self.args.data_dir, 'stdpose_rc_with_view_unit64.npy'))

        self._update_similarities()


    def _load_seq_from_df(self, pose_df):
        bad_columns = [col for col in pose_df.columns if col not in self.translation_mappings]
        assert bad_columns == [], f'Some columns are not in the translation mappings{bad_columns}'
        pose_df.columns = [self.translation_mappings[col] for col in pose_df.columns.values if col in self.translation_mappings]
        annot = pose_df_to_dict(pose_df)
        seq = annotations2motion(self.config.unique_nr_joints, annot['annotations'], scale=1)
        seq = preprocess_sequence(seq)
        return seq
    

    def _adjust_vid_length(self):
        s1_len = self.vid1.shape[0]
        s2_len = self.vid2.shape[0]

        ratio = s1_len / s2_len

        if ratio > 1.3:
            self.vid1 = self.vid1.iloc[::2, :]
            self.vid1.reset_index(drop=True, inplace=True)
            self.vid1['Frame'] = self.vid1.index
        elif ratio < 0.7:
            self.vid2 = self.vid2.iloc[::2, :]
            self.vid2.reset_index(drop=True, inplace=True)
            self.vid2['Frame'] = self.vid2.index


    def set_window_size(self, window_size):
        '''
        Sets the window size for the similarity prediction.
        
        Parameters:
        window_size: int
            Window size used for similarity prediction.
        '''
        self.args.video_sampling_window_size = window_size
        self._update_similarities()

    
    def set_stride_size(self, stride_size):
        '''
        Sets the stride size for the similarity prediction.
        
        Parameters:
        stride_size: int
            Stride size used for similarity prediction.
        '''
        self.args.video_sampling_stride = stride_size
        self._update_similarities()

    
    def set_window_stride_size(self, window_size, stride_size):
        '''
        Sets the window and stride size for the similarity prediction.
        
        Parameters:
        window_size: int
            Window size used for similarity prediction.
        stride_size: int
            Stride size used for similarity prediction.
        '''
        self.args.video_sampling_window_size = window_size
        self.args.video_sampling_stride = stride_size
        self._update_similarities()


    def get_embeddings(self, seq, window_size, stride):

        seq = preprocess_motion2d_rc(seq, self.mean_pose_bpe, self.std_pose_bpe, use_all_joints_on_each_bp=self.args.use_all_joints_on_each_bp)
        seq = seq.to(self.config.device)
        seq_features = self.similarity_analyzer.get_embeddings(seq, video_window_size=window_size, video_stride=stride)
        
        return seq_features
    

    def _update_similarities(self):

        seq1_features = self.get_embeddings(self.seq1, self.args.video_sampling_window_size, self.args.video_sampling_stride)
        seq2_features = self.get_embeddings(self.seq2, self.args.video_sampling_window_size, self.args.video_sampling_stride)
        
        motion_similarity_per_window = \
            self.similarity_analyzer.get_similarity_score(seq1_features, seq2_features, similarity_window_size=self.args.similarity_measurement_window_size)

        self.motion_similarity_per_window = motion_similarity_per_window

    
    def draw_frame(self, seq, frame_idx, width=1920, height=1080):
        '''
        Draws the frame in the sequence at frame_idx.

        Parameters:
        seq: np.ndarray
            The sequence of poses.
        frame_idx: int
            The index of the frame to draw.
        width: int, default=1920
            Width of the output frame.
        height: int, default=1080
            Height of the output frame.

        Returns:
        np.ndarray
            The frame with the stick figure drawn on it. To display it, use matplotlib.pyplot.imshow(frame).
        '''
        
        total_seq_len = seq.shape[-1] 
        percentage_processed = frame_idx / total_seq_len
        frame = self._scale_sequence(seq, width, height)[:, :, frame_idx]
        canvas = np.ones((height, width, 3), np.uint8) * 0

        color_per_joint = get_colors_per_joint(self.motion_similarity_per_window, percentage_processed, self.args.thresh)

        draw_seq(canvas, frame, color_per_joint, is_connected_joints=self.args.connected_joints)

        return canvas
    

    def _get_video_frames(self, seq, filename, width=1920, height=1080, include_score=False, memory=None):
        all_frames = []
        total_canvas = np.ones((height, width, 3), np.uint8) * 0

        for i, _ in enumerate(seq.transpose(2, 0, 1)):
            canvas = self.draw_frame(seq, i, width, height)

            put_filename_in_video(canvas, filename)
            
            if memory is not None:
                if i % memory == 1:
                    total_canvas += canvas

            if include_score:
                put_similarity_score_in_video(canvas, self.motion_similarity_per_window, i / seq.shape[-1], self.args.thresh)

            all_frames.append(total_canvas + canvas)

        return all_frames


    def _scale_sequence(self, seq, maxw=1920, maxh=1080, buffer=50):
        ratio = min((maxw - buffer) / seq[:, 0, :].max(), (maxh - buffer) / seq[:, 1, :].max())
        return seq * ratio
    
    
    def create_stick_figure_comparison_video(self, out_path, fps=30, out_width=1920, out_height=1080, memory=None):
        '''
        Creates a video comparing the two sequences.
        
        Parameters:
        out_path: str
            Path to the output video file.
        fps: int, default=30
            Frames per second.
        out_width: int, default=1920
            Width of the output video.
        out_height: int, default=1080
            Height of the output video. Final video will be 2x this height, as it will contain two sequences on top of each.
        memory: int, default=None
            If not None, the video will be created in memory, and the frames will be added to the total canvas every memory-th frame.
        '''
        total_vid_len = min(self.seq1.shape[-1], self.seq2.shape[-1])

        s1 = self.seq1[:, :, :total_vid_len]
        s2 = self.seq2[:, :, :total_vid_len]

        s1 = self._get_video_frames(s1, filename=self.vid1_pose_file, width=out_width, height=out_height, include_score=True, memory=memory)
        s2 = self._get_video_frames(s2, filename=self.vid2_pose_file, width=out_width, height=out_height, memory=memory)

        all_frames = np.concatenate((s1, s2), axis=1)

        io.mimwrite(out_path, all_frames, fps=fps)


    def show_analysed_frames(self, seq_length=None, window_size=None, stride_size=None):
        '''
        Prints the ranges of frames that were analysed.

        Parameters:
        seq_length: int, default=None
            Length of the sequence. If None, will use the length of the first sequence.
        window_size: int, default=None
            Window size used for similarity prediction. If None, will use the window size currently set in the class.
        stride_size: int, default=None
            Stride size used for similarity prediction. If None, will use the stride currently set in the class.
        
        Example:
        >>> show_analysed_frames(60, 30, 15)
        Range =  3
        From: 0  To:  30
        From: 15  To:  45
        From: 30  To:  60
        '''
        if seq_length is None:
            seq_length = self.seq1.shape[-1]
        if window_size is None:
            window_size = self.args.video_sampling_window_size
        if stride_size is None:
            stride_size = self.args.video_sampling_stride

        print('Range = ', math.ceil(((seq_length - window_size + 1) / stride_size)))
        for i in range(math.ceil(((seq_length - window_size + 1) / stride_size))):
            print('From:', i * stride_size, ' To: ', i * stride_size + window_size)


class VolleyballAttackAnalyser(VolleyballAnalyser):
    def __init__(self, vid1_pose_file, vid2_pose_file, window_size=30, stride=30, translation_mappings_path='attack_pose_data/translation_mappings.json', num_attack_frames=30):
        super().__init__(vid1_pose_file, vid2_pose_file, window_size, stride, translation_mappings_path)
        '''
        Constructs all the necessary attributes for the VolleyballAnalyser object.
        
        Parameters:
        -----------
        vid1_pose_file: str
            Path to the first video's pose excel file.
        vid2_pose_file: str
            Path to the second video's pose excel file.
        window_size: int, default=30
            Window size used for similarity prediction.
        stride: int, default=30
            Stride size used for similarity prediction.
        translation_mappings_path: str, default='attack_pose_data/translation_mappings.json'
            Path to the translation mappings file.
        num_attack_frames: int, default=30
            Number of frames around the attack to cut the sequences to.
        '''
        self.num_attack_frames = num_attack_frames
        self._get_attack_frame_numbers()
        self._cut_sequences_around_attack()


    def _get_attack_frame_numbers(self):
        assert 'phase' in self.vid1.columns, 'No "phase" column in the first pose file'
        assert 'phase' in self.vid2.columns, 'No "phase" column in the second pose file'
        a1 = int(self.vid1[self.vid1.phase == 'attack'].frame_number.values.mean())
        a2 = int(self.vid2[self.vid2.phase == 'attack'].frame_number.values.mean())
        self.attack_frames = [a1, a2]


    def _cut_sequences_around_attack(self):
        around = self.num_attack_frames // 2
        self.seq1 = self.seq1[:, :, self.attack_frames[0] - around: self.attack_frames[0] + around]
        self.seq2 = self.seq2[:, :, self.attack_frames[1] - around: self.attack_frames[1] + around]
        self.args.video_sampling_window_size = min(self.args.video_sampling_window_size, self.num_attack_frames, self.seq1.shape[-1])
        self.args.video_sampling_stride = min(self.args.video_sampling_stride, self.num_attack_frames, self.seq1.shape[-1])
        self._update_similarities()


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


class VideoGenerator:
    translation_mappings = json.load(open('attack_pose_data/translation_mappings.json'))
    colors_per_joint = np.array([
        [255, 255, 255],  # nose
        [255, 255, 255],  # neck
        [0, 0, 255],  # right shoulder
        [0, 0, 255],  # right elbow
        [0, 0, 255],  # right wrist
        [255, 0, 0],  # left shoulder
        [255, 0, 0],  # left elbow
        [255, 0, 0],  # left wrist
        [255, 255, 255],  # mid hip
        [0, 0, 255],  # right hip
        [0, 0, 255],  # right knee
        [0, 0, 255],  # right ankle
        [255, 0, 0],  # left hip
        [255, 0, 0],  # left knee
        [255, 0, 0],  # left ankle
    ])


    def __init__(self, pose_filename) -> None:
        self.pose_filename = pose_filename
        self.pose_data = self._read_pose_excel()
        self.pose_seq = self._load_seq_from_df(self.pose_data)

    def _read_pose_excel(self):
        return pd.read_excel(self.pose_filename, index_col=0)
    
    def _load_seq_from_df(self, pose_df):
        bad_columns = [col for col in pose_df.columns if col not in self.translation_mappings]
        assert bad_columns == [], f'Some columns are not in the translation mappings{bad_columns}'
        pose_df.columns = [self.translation_mappings[col] for col in pose_df.columns.values if col in self.translation_mappings]
        annot = pose_df_to_dict(pose_df)
        seq = annotations2motion(15, annot['annotations'], scale=1)
        seq = preprocess_sequence(seq)
        return seq
    
    def _get_frame(self, frame_idx):
        return self.pose_data[self.pose_data.frame_number == frame_idx]

    def _scale_sequence(self, seq, maxw=1920, maxh=1080, buffer=50):
        ratio = min((maxw - buffer) / seq[:, 0, :].max(), (maxh - buffer) / seq[:, 1, :].max())
        return seq * ratio
    
    def _draw_frame(self, frame_idx, width=1920, height=1080):
        '''
        Draws the frame in the sequence at frame_idx.

        Parameters:
        seq: np.ndarray
            The sequence of poses.
        frame_idx: int
            The index of the frame to draw.
        width: int, default=1920
            Width of the output frame.
        height: int, default=1080
            Height of the output frame.

        Returns:
        np.ndarray
            The frame with the stick figure drawn on it. To display it, use matplotlib.pyplot.imshow(frame).
        '''
        seq = self.pose_seq
        frame = self._scale_sequence(seq, width, height)[:, :, frame_idx]
        canvas = np.ones((height, width, 3), np.uint8) * 0

        draw_seq(canvas, frame, self.colors_per_joint, is_connected_joints=True)

        return canvas
    
    def show_frame(self, frame_idx, width=1920, height=1080):
        '''
        Shows the frame with the stick figure drawn on it.

        Parameters:
        frame_idx: int
            The index of the frame to draw.
        width: int, default=1920
            Width of the output frame.
        height: int, default=1080
            Height of the output frame.
        '''
        frame = self._draw_frame(frame_idx, width, height)
        plt.imshow(frame)
        plt.show()
    
    def _get_video_frames(self, filename, width=1920, height=1080, memory=None):
        seq = self.pose_seq
        all_frames = []
        total_canvas = np.ones((height, width, 3), np.uint8) * 0

        for i, _ in enumerate(seq.transpose(2, 0, 1)):
            canvas = self._draw_frame(i, width, height)

            put_filename_in_video(canvas, filename)
            
            if memory is not None:
                if i % memory == 1:
                    total_canvas += canvas

            all_frames.append(total_canvas + canvas)

        return all_frames
    
    def create_stick_video(self, out_file, fps=30, width=1920, height=1080, memory=None):
        s = self._get_video_frames(filename=self.pose_filename, width=width, height=height, memory=memory)
        io.mimwrite(out_file, s, fps=fps)




DATA_DIR = '../bpe/attack_pose_data/'
EXCEL_DATA_DIR = '../bpe/attack_pose_data/excel_files/'
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


def normalize_keypoints_relative_to_bbox(df, x_col, y_col):
    """
    Normalize keypoints to a [0, 1] range relative to the bounding box encompassing all keypoints.

    Parameters:
    - df: pandas DataFrame with 'x' and 'y' columns for keypoint coordinates.

    Returns:
    - df with added 'x_normalized' and 'y_normalized' columns.
    """
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    x = (df[x_col] - x_min) / (x_max - x_min)
    y = (df[y_col] - y_min) / (y_max - y_min)
    
    return x, y


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


def load_pose_data(pose_df_path, mappings):
    df = pd.read_excel(pose_df_path, index_col=0)
    df.rename(columns=mappings, inplace=True)
    fps = df['fps'].values[0]
    return df, fps


def get_normalised_kepoints(aggregated_pose_df, joints):
    norm_df = bbox_normalize_joint_coordinates(aggregated_pose_df[[joint + '_x' for joint in joints] + [joint + '_y' for joint in joints]].copy())
    normalised_keypoints_y = np.array([norm_df[joint + '_y_normalized'].values for joint in joints])
    normalised_keypoints_x = np.array([norm_df[joint + '_x_normalized'].values for joint in joints])
    normalised_keypoints_all = np.concatenate([normalised_keypoints_x, normalised_keypoints_y], axis=0)
    return normalised_keypoints_all.T, normalised_keypoints_x.T, normalised_keypoints_y.T


def get_breakpoints(normalised_keypoints, num_phases):
    bkpts_algorithm = rpt.KernelCPD(kernel="rbf", min_size=14)
    bkpts = bkpts_algorithm.fit_predict(normalised_keypoints, n_bkps=num_phases - 1)
    return bkpts


def preprocess_df_for_bpe(aggregated_pose_df, config):
    annot = pose_df_to_dict(aggregated_pose_df)
    seq = annotations2motion(config.unique_nr_joints, annot['annotations'], scale=1)
    seq = preprocess_sequence(seq)
    return seq


def get_embeddings(seq, window_size, stride, similarity_analyzer, mean_pose_bpe, std_pose_bpe, config, args):
    seq = preprocess_motion2d_rc(seq, mean_pose_bpe, std_pose_bpe, use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)
    seq = seq.to(config.device)
    seq_features = similarity_analyzer.get_embeddings(seq, video_window_size=window_size, video_stride=stride)
    return seq_features


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


def flatten_results(results):
    rows = []
    for filename, phases in results.items():
        row = {'filename': filename}
        for phase, attributes in phases.items():
            for key, value in attributes.items():
                row[f'{phase}_{key}'] = value
        rows.append(row)
    return rows


def reindex(bkpts, index):
    result_reindexed = []
    for bkp in bkpts:
        for k,v in index.items():
            if bkp == v:
                result_reindexed.append(k)
                break
    return result_reindexed


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