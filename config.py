# Brycen Westgarth and Tristan Jogminas
# March 5, 2021
# Modified by Marcin Zatorski, 15.09.2022
import argparse

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Neural style transfer for videos')
        parser.add_argument('-i', '--input', required=True, help='Path to input video file')
        parser.add_argument('-o', '--output', required=True, help='Path to output video file')
        parser.add_argument('--input-fps', type=int, default=30)
        parser.add_argument('--output-fps', type=int, default=30)
        parser.add_argument('--frame-height', type=int, default=360, help='Height of output frame')
        parser.add_argument('--style-sequence', type=int, nargs='+', default=[0, 1, 2])
        parser.add_argument('--ghost-frame-transparency', type=float, default=0.1)
        parser.add_argument('--preserve-colors', action='store_true')
        args = parser.parse_args()

        # defines the maximum height dimension in pixels. Used for down-sampling the video frames
        self.FRAME_HEIGHT = args.frame_height
        # defines the rate at which you want to capture frames from the input video
        self.INPUT_FPS = args.input_fps
        self.INPUT_VIDEO_PATH = args.input

        self.STYLE_REF_DIRECTORY = './style_ref'
        # defines the reference style image transition sequence. Values correspond to indices in STYLE_REF_DIRECTORY
        # add None in the sequence to NOT apply style transfer for part of the video (ie. [None, 0, 1, 2])  
        self.STYLE_SEQUENCE = args.style_sequence

        self.OUTPUT_FPS = args.output_fps
        self.OUTPUT_VIDEO_PATH = args.output

        self.GHOST_FRAME_TRANSPARENCY = args.ghost_frame_transparency
        self.PRESERVE_COLORS = args.preserve_colors

        self.TENSORFLOW_CACHE_DIRECTORY = f'./tensorflow_cache'
        self.TENSORFLOW_HUB_HANDLE = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
