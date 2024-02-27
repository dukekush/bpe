import os 
import math
import numpy as np
import pandas as pd
import argparse
import os
import math
import imageio as io

os.chdir('/Users/jniedziela/Developer/master/bpe/')

from bpe import Config
from bpe.functional.motion import preprocess_motion2d_rc, cocopose2motion
from bpe.similarity_analyzer import SimilarityAnalyzer
from bpe.functional.visualization import get_colors_per_joint, put_similarity_score_in_video, preprocess_sequence, draw_seq, draw_frame


DATA_DIR = 'model_data/'
MODEL_PATH = 'model_data/pretrained_model/model/model_best.pth.tar'


def config_parser():
    video1 = ''
    video2 = ''
    vid1_json_dir = ''
    vid2_json_dir = ''
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
    parser.add_argument('-v1', '--vid1_json_dir', type=str, required=True, help="video1's coco annotation json")
    parser.add_argument('-v2', '--vid2_json_dir', type=str, required=True, help="video2's coco annotation json")
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
        '--data_dir', DATA_DIR, 
        '--model_path', MODEL_PATH,
        '--video1', video1,
        '--video2', video2,
        '-v1', vid1_json_dir,
        '-v2', vid2_json_dir,
        '--img1_height', str(img1_height),
        '--img1_width', str(img1_width),
        '--img2_height', str(img2_height),
        '--img2_width', str(img2_width),
        ])
    
    return args


class VolleyWrapper:
    def __init__(self, vid1, vid2, window_size=30, stride=30):
        self.args = config_parser()
        self.args.vid1_json_dir = vid1
        self.args.vid2_json_dir = vid2
        self.args.video_sampling_window_size = window_size
        self.args.video_sampling_stride = stride
        
        self.config = Config(self.args)
        self.similarity_analyzer = SimilarityAnalyzer(self.config, self.args.model_path)

        self.mean_pose_bpe = np.load(os.path.join(DATA_DIR, 'meanpose_rc_with_view_unit64.npy'))
        self.std_pose_bpe = np.load(os.path.join(DATA_DIR, 'stdpose_rc_with_view_unit64.npy'))
        
        self.seq1, self.seq2 = self._load_seq_json()

        self.motion_similarity_per_window = self._update_similarities()


    def set_window_size(self, window_size):
        self.args.video_sampling_window_size = window_size
        self.motion_similarity_per_window = self._update_similarities()

    
    def set_stride_size(self, stride_size):
        self.args.video_sampling_stride = stride_size
        self.motion_similarity_per_window = self._update_similarities()

    
    def set_window_stride_size(self, window_size, stride_size):
        self.args.video_sampling_window_size = window_size
        self.args.video_sampling_stride = stride_size
        self.motion_similarity_per_window = self._update_similarities()


    def embeddings(self, seq, window_size, stride):

        seq = preprocess_motion2d_rc(seq, self.mean_pose_bpe, self.std_pose_bpe,
                                    invisibility_augmentation=self.args.use_invisibility_aug,
                                    use_all_joints_on_each_bp=self.args.use_all_joints_on_each_bp)

        seq = seq.to(self.config.device)

        # get embeddings
        seq_features = self.similarity_analyzer.get_embeddings(seq, video_window_size=window_size,
                                                            video_stride=stride)
        
        return seq_features
    

    def _update_similarities(self):

        seq1_origin = preprocess_motion2d_rc(self.seq1, self.mean_pose_bpe, self.std_pose_bpe,
                                            invisibility_augmentation=self.args.use_invisibility_aug,
                                            use_all_joints_on_each_bp=self.args.use_all_joints_on_each_bp)
        seq2_origin = preprocess_motion2d_rc(self.seq2, self.mean_pose_bpe, self.std_pose_bpe,
                                                invisibility_augmentation=self.args.use_invisibility_aug,
                                                use_all_joints_on_each_bp=self.args.use_all_joints_on_each_bp)

        seq1_origin = seq1_origin.to(self.config.device)
        seq2_origin = seq2_origin.to(self.config.device)

        # get embeddings
        seq1_features = self.similarity_analyzer.get_embeddings(seq1_origin, video_window_size=self.args.video_sampling_window_size,
                                                            video_stride=self.args.video_sampling_stride)
        seq2_features = self.similarity_analyzer.get_embeddings(seq2_origin, video_window_size=self.args.video_sampling_window_size,
                                                            video_stride=self.args.video_sampling_stride)
        
        # get motion similarity
        motion_similarity_per_window = \
            self.similarity_analyzer.get_similarity_score(seq1_features, seq2_features,
                                                        similarity_window_size=self.args.similarity_measurement_window_size)

        self.motion_similarity_per_window = motion_similarity_per_window

        return motion_similarity_per_window
    
    def _load_seq_json(self):
        s1 = cocopose2motion(self.config.unique_nr_joints, self.args.vid1_json_dir, scale=1,
                                visibility=self.args.use_invisibility_aug)
        s2 = cocopose2motion(self.config.unique_nr_joints, self.args.vid2_json_dir, scale=1,
                                visibility=self.args.use_invisibility_aug)[:, :, self.args.pad2:]
        
        s1 = preprocess_sequence(s1)
        s2 = preprocess_sequence(s2)
        
        return s1, s2

    
    def draw_frame(self, seq, frame_idx, width=1920, height=1080):
        '''
        Draws the frame in the sequence at frame_idx.
        If motion similarity is calculated, draws the joints in different colors depending on the similarity.
        '''
        assert self.motion_similarity_per_window is not None, 'motion similarity is not calculated yet, run get_similarities(s1, s2) first'
        total_vid_len = seq.shape[-1]  # currently frame num is last shape, in original input is transposed
        percentage_processed = frame_idx / total_vid_len
        frame = seq[:, :, frame_idx]
        canvas = np.ones((height, width, 3), np.uint8) * 0

        # 15 * RGB: (15, 3) 
        color_per_joint = get_colors_per_joint(self.motion_similarity_per_window, percentage_processed, self.args.thresh)

        draw_seq(canvas, frame, color_per_joint, left_padding=0, is_connected_joints=self.args.connected_joints)

        return canvas
    
    def _get_video_frames(self, seq):
        all_frames = []
        # total_canvas = np.ones((1080, 1920, 3), np.uint8) * 0

        for i, _ in enumerate(seq.transpose(2, 0, 1)):
            canvas = self.draw_frame(seq, i)
            put_similarity_score_in_video(canvas, self.motion_similarity_per_window, i / seq.shape[-1], self.args.thresh)
            all_frames.append(canvas)

            # if i % 30 == 0:
            #     total_canvas += canvas
            #     all_frames.append(total_canvas)
            # else:
            #     all_frames.append(total_canvas + canvas)

        return all_frames
    

    def show_analysed_frames(self, seq_length=None, window_size=None, stride_size=None):
        '''
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
    

    def _scale_sequence(self, seq, maxw=1920, maxh=1080, buffer=50):
        ratio = min((maxw - buffer) / seq[:, 0, :].max(), (maxh - buffer) / seq[:, 1, :].max())
        return seq * ratio


    def draw_two_sequences(self, out_path, fps=30):
        total_vid_len = min(self.seq1.shape[-1], self.seq2.shape[-1])

        s1 = self._scale_sequence(self.seq1[:, :, :total_vid_len])
        s2 = self._scale_sequence(self.seq2[:, :, :total_vid_len])

        s1 = self._get_video_frames(s1)
        s2 = self._get_video_frames(s2)

        all_frames = np.concatenate((s1, s2), axis=1)

        io.mimwrite(out_path, all_frames, fps=fps)


# Other function for analysis
def aggregate_similarities(similarities_list, aggregation_func):
    result = {}
    count = len(similarities_list)

    if count == 0:
        return result

    for key in similarities_list[0].keys():
        values = [item[key] for item in similarities_list]
        result[key] = aggregation_func(values)

    return result


def aggregate_similarities_all(similarities_list):
    result = {}
    count = len(similarities_list)
    
    if count == 0:
        return result
    
    for aggfunc in [np.min, np.max, np.mean, np.std]:
        for key in similarities_list[0].keys():
            values = [item[key] for item in similarities_list]
            result[key + '_' + aggfunc.__name__] = aggfunc(values)

    return result


def show_analysed_frames(seq_length, window_size, stride_size):
    '''
    >>> show_analysed_frames(60, 30, 15)
    Range =  3
    From: 0  To:  30
    From: 15  To:  45
    From: 30  To:  60
    '''
    print('Range = ', math.ceil(((seq_length - window_size + 1) / stride_size)))
    for i in range(math.ceil(((seq_length - window_size + 1) / stride_size))):
        # seq[:, :, i * stride: i * stride + window_size]
        print('From:', i * stride_size, ' To: ', i * stride_size + window_size)


def adjusted_r2(r2, n, p):
    """
    Calculate adjusted R-squared.

    Parameters:
        r2 (float): R-squared value.
        n (int): Number of samples.
        p (int): Number of features.

    Returns:
        float: Adjusted R-squared value.
    """
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


def get_attack_frames_per_video(excel_data_dir): 
    # get frame in which attack happens (ball hit)
    attack_frames = {}
    for file in os.listdir(excel_data_dir):
        df = pd.read_excel(excel_data_dir + file)
        attack_frames[file.replace('.xlsx', '.json')] = df[df.phase == 'attack'][1:-1].Frame.iloc[0]
    return attack_frames