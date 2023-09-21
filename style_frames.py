# Brycen Westgarth and Tristan Jogminas
# March 5, 2021
# Modified by Marcin Zatorski, 15.09.2022
# Modified by Niccolò Perego, 18.09.2023

from config import Config
from moviepy.editor import *
from tqdm import tqdm
import logging
import cv2
import glob
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.config.optimizer.set_jit('autoclustering')


class StyleFrame:

    MAX_CHANNEL_INTENSITY = 255.0

    def __init__(self, conf=Config):
        self.conf = conf()
        os.environ['TFHUB_CACHE_DIR'] = self.conf.TENSORFLOW_CACHE_DIRECTORY
        self.hub_module = hub.load(self.conf.TENSORFLOW_HUB_HANDLE)
        self.style_directory = glob.glob(f'{self.conf.STYLE_REF_DIRECTORY}/*')
        self.ref_count = len(self.conf.STYLE_SEQUENCE)

        # Init input related variables
        self.video_capture = self.create_video_capture()

    def create_video_capture(self):
        vid_obj = cv2.VideoCapture(self.conf.INPUT_VIDEO_PATH)
        success, image = vid_obj.read()
        if image is None:
            raise ValueError(
                f'ERROR: Please provide missing video: {self.conf.INPUT_VIDEO_PATH}')

        # Set frame width and frame length
        scale_constant = (self.conf.FRAME_HEIGHT / image.shape[0])
        self.frame_width = int(image.shape[1] * scale_constant)

        frame_count = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vid_obj.get(cv2.CAP_PROP_FPS)
        # new number of frames
        self.frame_length = int(frame_count / fps * self.conf.INPUT_FPS)

        return vid_obj

    def get_style_info(self):
        style_refs = list()
        resized_ref = False
        style_files = sorted(self.style_directory)
        self.t_const = self.frame_length if self.ref_count == 1 else np.ceil(
            self.frame_length / (self.ref_count - 1))

        # Open first style ref and force all other style refs to match size
        first_style_ref = cv2.imread(style_files.pop(0))
        first_style_ref = cv2.cvtColor(first_style_ref, cv2.COLOR_BGR2RGB)
        first_style_height, first_style_width, _rgb = first_style_ref.shape
        style_refs.append(first_style_ref / self.MAX_CHANNEL_INTENSITY)

        for filename in style_files:
            style_ref = cv2.imread(filename)
            style_ref = cv2.cvtColor(style_ref, cv2.COLOR_BGR2RGB)
            style_ref_height, style_ref_width, _rgb = style_ref.shape
            # Resize all style_ref images to match first style_ref dimensions
            if style_ref_width != first_style_width or style_ref_height != first_style_height:
                resized_ref = True
                style_ref = cv2.resize(
                    style_ref, (first_style_width, first_style_height))
            style_refs.append(style_ref / self.MAX_CHANNEL_INTENSITY)

        if resized_ref:
            print('WARNING: Resizing style images which may cause distortion. To avoid this, please provide style images with the same dimensions')

        self.transition_style_seq = list()
        for i in range(self.ref_count):
            if self.conf.STYLE_SEQUENCE[i] is None:
                self.transition_style_seq.append(None)
            else:
                self.transition_style_seq.append(
                    style_refs[self.conf.STYLE_SEQUENCE[i]])

    def _trim_img(self, img):
        return img[:self.conf.FRAME_HEIGHT, :self.frame_width]

    def get_output_frames(self):
        ghost_frame = None

        video_writer = self.create_video_writer()
        frame_interval = (1.0 / self.conf.INPUT_FPS) * 1000

        halfway_frame = np.ceil(self.frame_length / 2)

        count = 0
        success = True
        progress_bar = tqdm(total=self.frame_length)
        while success:
            # where we at
            msec_timestamp = count * frame_interval
            # set reader to time and read
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC, msec_timestamp)
            success, content_img = self.video_capture.read()
            if not success:
                break
            # prep
            content_img = cv2.resize(
                content_img, (self.frame_width, self.conf.FRAME_HEIGHT))
            content_img = cv2.cvtColor(
                content_img, cv2.COLOR_BGR2RGB) / self.MAX_CHANNEL_INTENSITY

            # first tests
            # curr_style_img_index = int(count / self.t_const)
            # mix_ratio = 1 - ((count % self.t_const) / self.t_const)
            # inv_mix_ratio = 1 - mix_ratio

            curr_style_img_index = 0 if count < halfway_frame else 1
            mix_ratio = 1
            inv_mix_ratio = 0

            prev_image = self.transition_style_seq[curr_style_img_index] if curr_style_img_index < self.ref_count else None
            next_image = self.transition_style_seq[curr_style_img_index +
                                                   1] if curr_style_img_index + 1 < self.ref_count else None

            prev_is_content_img = False
            next_is_content_img = False
            if prev_image is None:
                prev_image = content_img
                prev_is_content_img = True
            if next_image is None:
                next_image = content_img
                next_is_content_img = True
            # If both, don't need to apply style transfer
            if prev_is_content_img and next_is_content_img:
                temp_ghost_frame = cv2.cvtColor(
                    ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
                video_writer.write(temp_ghost_frame.astype(np.uint8))
                continue

            if count > 0:
                content_img = ((1 - self.conf.GHOST_FRAME_TRANSPARENCY) *
                               content_img) + (self.conf.GHOST_FRAME_TRANSPARENCY * ghost_frame)
            content_img = tf.cast(
                tf.convert_to_tensor(content_img), tf.float32)

            if prev_is_content_img:
                blended_img = next_image
            elif next_is_content_img:
                blended_img = prev_image
            else:
                prev_style = mix_ratio * prev_image
                next_style = inv_mix_ratio * next_image
                blended_img = prev_style + next_style

            blended_img = tf.cast(
                tf.convert_to_tensor(blended_img), tf.float32)
            expanded_blended_img = tf.constant(
                tf.expand_dims(blended_img, axis=0))
            expanded_content_img = tf.constant(
                tf.expand_dims(content_img, axis=0))
            # Apply style transfer
            stylized_img = self.hub_module(
                expanded_content_img, expanded_blended_img).pop()
            stylized_img = tf.squeeze(stylized_img)

            # Re-blend
            if prev_is_content_img:
                prev_style = mix_ratio * content_img
                next_style = inv_mix_ratio * stylized_img
            if next_is_content_img:
                prev_style = mix_ratio * stylized_img
                next_style = inv_mix_ratio * content_img
            if prev_is_content_img or next_is_content_img:
                stylized_img = self._trim_img(
                    prev_style) + self._trim_img(next_style)

            if self.conf.PRESERVE_COLORS:
                stylized_img = self._color_correct_to_input(
                    content_img, stylized_img)

            ghost_frame = np.asarray(self._trim_img(stylized_img))

            temp_ghost_frame = cv2.cvtColor(
                ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
            video_writer.write(temp_ghost_frame.astype(np.uint8))
            progress_bar.update(1)
            count += 1

        video_writer.release()
        self.video_capture.release()

    def _color_correct_to_input(self, content, generated):
        # image manipulations for compatibility with opencv
        content = np.array(
            (content * self.MAX_CHANNEL_INTENSITY), dtype=np.float32)
        content = cv2.cvtColor(content, cv2.COLOR_BGR2YCR_CB)
        generated = np.array(
            (generated * self.MAX_CHANNEL_INTENSITY), dtype=np.float32)
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2YCR_CB)
        generated = self._trim_img(generated)
        # extract channels, merge intensity and color spaces
        color_corrected = np.zeros(generated.shape, dtype=np.float32)
        color_corrected[:, :, 0] = generated[:, :, 0]
        color_corrected[:, :, 1] = content[:, :, 1]
        color_corrected[:, :, 2] = content[:, :, 2]
        return cv2.cvtColor(color_corrected, cv2.COLOR_YCrCb2BGR) / self.MAX_CHANNEL_INTENSITY

    def create_video_writer(self):
        # Use H.264 encoding to make videos about 2-3 times smaller
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(self.conf.NO_AUDIO_OUTPUT_VIDEO_PATH, fourcc,
                                       self.conf.OUTPUT_FPS, (self.frame_width, self.conf.FRAME_HEIGHT))
        if not video_writer.isOpened():
            # Fallback to mp4v if, for example, opencv was installed through pip
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(self.conf.NO_AUDIO_OUTPUT_VIDEO_PATH, fourcc,
                                           self.conf.OUTPUT_FPS, (self.frame_width, self.conf.FRAME_HEIGHT))
        return video_writer

    def detach_audio(self):
        video = VideoFileClip(self.conf.INPUT_VIDEO_PATH)
        audio = video.audio
        audio.write_audiofile(self.conf.OUTPUT_AUDIO_PATH)

    def reattach_audio(self):
        styled_video_clip = VideoFileClip(self.conf.NO_AUDIO_OUTPUT_VIDEO_PATH)
        audio_clip = AudioFileClip(self.conf.OUTPUT_AUDIO_PATH)
        video_with_audio = styled_video_clip.set_audio(audio_clip)
        video_with_audio.write_videofile(
            self.conf.COMPLETE_OUTPUT_VIDEO_PATH)

    def run(self):
        print('Detatching audio')
        self.detach_audio()
        print('Getting style info')
        self.get_style_info()
        print('Doing style transfer')
        self.get_output_frames()
        print('Reattaching audio')
        self.reattach_audio()


if __name__ == '__main__':
    StyleFrame().run()
