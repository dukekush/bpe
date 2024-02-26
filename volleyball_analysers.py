import os 
import math
import numpy as np
import pandas as pd
import argparse
import os
import math
import imageio as io
import json

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
        elif ratio < 0.7:
            self.vid2 = self.vid2.iloc[::2, :]
            self.vid2.reset_index(drop=True, inplace=True)


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
        a1 = int(self.vid1[self.vid1.phase == 'attack'].index.values.mean())
        a2 = int(self.vid2[self.vid2.phase == 'attack'].index.values.mean())
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