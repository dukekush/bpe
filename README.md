# Pose estimation analyses project with volleyball data


## Folders info
### bin
        BPE scripts for training, inference and outputs.
### bpe
        BPE model codes, settings, and helper functions.
### model_data
        pretrained BPE model, files necessary for model.
### development
        notebooks used in development of phase comparison, development of functions for data preprocessing. Also contains notebook for final analysis of breakpoints.

## Important files info
### final_comparisons.ipynb
        Creates comparison videos for pair of files.
### create_stick_videos.ipynb
        Creates skeleton videos based on pose data.
### phases_process_scores.ipynb
        Calculates comparison scores of reference with comparison videos and outputs an excel file with scores.
### phases_to_photo_sequence.ipynb
        Creates colour pallete and sequence pictures, illustrating system's outputs.
### phases_to_video_RANDOM_30.ipynb 
        Creates output videos, including phases annotations and scores based on BPE model, given input blurred videos and files with keypoints.
### volleyball_analysers.py
        A file containing necessary functions for notebooks and scripts.

<!-- ## Getting started
### Installation
- Clone the repository:
        
        git clone https://github.com/dukekush/volleyball_pose_bpe.git

- Go to the project directory:

        cd volleyball_pose_bpe

- Install dependencies:

        pip install -r requirements.txt

- Adjust the <b>PROJECT_GLOBAL_PATH</b> in <b>setup.py</b>:

        
        PROJECT_GLOBAL_PATH = '/absolute-path-to/volleyball_pose_bpe/'
        
        # In google colab:
        PROJECT_GLOBAL_PATH = '/content/volleyball_pose_bpe/'

### Creating videos
- Run examples in <b>video_comparison.ipynb</b> notebook. -->
