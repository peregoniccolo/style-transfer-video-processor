# Brycen Westgarth and Tristan Jogminas
# March 5, 2021
# Modified by Marcin Zatorski, 15.09.2022
# Modified by Niccolò Perego, 21.09.2023

import argparse
import os


class Config:
    default_ts = [0.0, 0.5, 1.0]
    default_ss = [0, 1, 2]

    def __init__(self):
        args = self.run_parser()

        # defines the maximum height dimension in pixels. Used for down-sampling the video frames
        self.FRAME_HEIGHT = args.frame_height
        # defines the rate at which you want to capture frames from the input video
        self.FPS = args.fps
        self.INPUT_VIDEO_PATH = args.input
        _, self.INPUT_FILENAME = os.path.split(self.INPUT_VIDEO_PATH)
        self.INPUT_NAME, _ = os.path.splitext(self.INPUT_FILENAME)

        self.STYLE_REF_DIRECTORY = args.style_folder
        assert os.path.exists(self.STYLE_REF_DIRECTORY)
        self.MIDWAY_DIRECTORY = args.midway_folder
        assert os.path.exists(self.MIDWAY_DIRECTORY)
        # defines the reference style image transition sequence. Values correspond to indices in STYLE_REF_DIRECTORY
        # add None in the sequence to NOT apply style transfer for part of the video (ie. [None, 0, 1, 2])
        self.STYLE_SEQUENCE = args.style_sequence
        self.TIME_SEQUENCE = args.time_sequence

        self.checks_on_sequences()

        self.OUTPUT_NAME = args.output_name
        self.OUTPUT_NAME, orig_ext = os.path.splitext(self.OUTPUT_NAME)
        if orig_ext != "":
            print("dropped extention, not required")
        self.OUTPUT_NAME += ".mp4"

        self.OUTPUT_AUDIO_PATH = (
            f"{self.MIDWAY_DIRECTORY}/audio_from_{self.INPUT_NAME}.mp3"
        )
        self.NO_AUDIO_OUTPUT_VIDEO_PATH = (
            f"{self.MIDWAY_DIRECTORY}/no_audio_{self.OUTPUT_NAME}"
        )

        self.OUTPUT_DESTINATION = args.output_destination
        assert os.path.exists(
            self.OUTPUT_DESTINATION
        ), "specified destination does not exist"

        self.COMPLETE_OUTPUT_VIDEO_PATH = os.path.join(
            self.OUTPUT_DESTINATION, self.OUTPUT_NAME
        )

        self.GHOST_FRAME_TRANSPARENCY = args.ghost_frame_transparency
        self.PRESERVE_COLORS = args.preserve_colors

        self.TENSORFLOW_CACHE_DIRECTORY = "./tensorflow_cache"
        if args.local_module == None:
            print("Model from https")
            self.TENSORFLOW_HUB_HANDLE = (
                "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
            )
        else:
            assert os.path.exists(
                args.local_model
            ), "specified local model directory does not exist"
            print("Local model specified")
            self.TENSORFLOW_HUB_HANDLE = args.local_model

        self.NO_AUDIO = args.no_audio

    def run_parser(self):
        parser = argparse.ArgumentParser(description="Neural style transfer for videos")
        parser.add_argument(
            "-i", "--input", required=True, help="Path to input video file"
        )
        parser.add_argument(
            "-o",
            "--output_name",
            type=str,
            default="output",
            help="Name output video file (without extention)",
        )
        parser.add_argument(
            "-d",
            "--output_destination",
            type=str,
            default=".",
            help="Destination folder for the output video file",
        )
        parser.add_argument("--fps", type=int, default=30)
        # parser.add_argument('--output-fps', type=int, default=30)
        parser.add_argument(
            "--frame_height", type=int, default=360, help="Height of output frame"
        )
        parser.add_argument("-ss", "--style_sequence", type=int, nargs="+")
        parser.add_argument("-ts", "--time_sequence", type=float, nargs="+")
        parser.add_argument("--ghost_frame_transparency", type=float, default=0.1)
        parser.add_argument("--preserve_colors", action="store_true")
        parser.add_argument("--no_audio", action="store_true")
        parser.add_argument(
            "-l",
            "--local_module",
            type=str,
            default=None,
            help="Path to the local module",
        )
        parser.add_argument(
            "-sf",
            "--style_folder",
            type=str,
            default="style_ref",
            help="Folder containing the style images (named accordingly, eg. 00, 01, ...)",
        )
        parser.add_argument(
            "-mf",
            "--midway_folder",
            type=str,
            default="midway",
            help="Folder that will contain checkpoints like detached audio and video files.",
        )

        args = parser.parse_args()

        if args.time_sequence == None and args.style_sequence != None:
            parser.error("--style_sequence requires --time_sequence")
        if args.time_sequence != None and args.style_sequence == None:
            parser.error("--time_sequence requires --style_sequence")

        if args.time_sequence == None:
            # both none, setting default
            args.time_sequence = self.default_ts
            args.style_sequence = self.default_ss

        return args

    def check_bounds_and_values(self):
        count = 0
        for value in self.TIME_SEQUENCE:
            if count == 0:
                assert value == 0, "starting value in time sequence is not 0"
            if count == len(self.TIME_SEQUENCE) - 1:
                assert value == 1, "ending value in time sequence is not 1"
            assert isinstance(
                value, float
            ), f"value at position {count} in time sequence is not a number"
            if not count == 0:
                assert (
                    value > self.TIME_SEQUENCE[count - 1]
                ), "values in time sequence must be monotonically growing"
            count += 1

    def check_images_exist(self):
        # TODO
        return

    def checks_on_sequences(self):
        # they cannot be empty because of how argparse works
        assert len(self.TIME_SEQUENCE) == len(
            self.STYLE_SEQUENCE
        ), "time and style sequences have different lengths"
        self.check_bounds_and_values()
        self.check_images_exist()
