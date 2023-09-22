# Neural Style Transfer Transition Video Processing
By Brycen Westgarth and Tristan Jogminas
Modified by Marcin Zatorski and Niccolò Perego

## Description
This code extends the [neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer) 
image processing technique to video
by generating smooth transitions between a sequence of 
reference style images across video frames. The generated output 
video is a highly altered, artistic representation of the input
video consisting of constantly changing abstract patterns and colors
that emulate the original content of the video. The user's choice
of style reference images, style sequence order, and style sequence
length allow for infinite user experimentation and the creation of 
an endless range of artistically interesting videos.


## System Requirements
This algorithm is computationally intensive so I highly 
recommend optimizing its performance by installing drivers for 
[Tensorflow GPU support](https://www.tensorflow.org/install/gpu)
if you have access to a CUDA compatible GPU. Alternatively, you can
take advantage of the free GPU resources available through Google Colab Notebooks. 
Even with GPU acceleration, the program may take several minutes to render a video. 

[Colab Notebook Version](https://colab.research.google.com/drive/1ZjSvUv0Wqib6khaiqcBvRrI5GeSjFcOV?usp=sharing)

## Configuration
All configuration of the video properties and input/output file
locations can be set by the user in config.py 

Configurable Variable in config.py			         | Description
------------------------|------------
FRAME_HEIGHT    | Sets height dimension in pixels to resize the output video to. Video width will be calculated automatically to preserve aspect ratio. Low values will speed up processing time but reduce output video quality 
FPS 			    | Defines the rate at which frames are captured from the input video (and written for the output one)
INPUT_VIDEO_PATH     	| Path to input video file
STYLE_SEQUENCE     	| List that contains the indices corresponding to the image files in the 'style_ref' folder. Defines the reference style image transition sequence. Changes between images happen linearly w.r.t. the times specified in TIME_SEQUENCE
TIME_SEQUENCE     	| List that contains the times (relative format, 0.0 - 1.0) at which the current style is changed for the next one.
<!-- OUTPUT_FPS		    | Defines the frame rate of the output video -->
OUTPUT_NAME   | Filename of output video to be created  (without extention)
OUTPUT_DESTINATION   | Destination folder path for the output video
GHOST_FRAME_TRANSPARENCY | Proportional feedback constant for frame generation. Should be a value between 0 and 1. Affects the amount change that can occur between frames and the smoothness of the transitions. 
PRESERVE_COLORS      | If True the output video will preserve the colors of the input video. If  False the program will perform standard style transfer

**The user must find and place their own style reference images in the `style_ref` directory. 
 Style reference images can be
arbitrary size. For best results, try to use style reference images with similar dimensions
and aspect ratios. Three example style reference images are given.**<br/>
<br/>
<!-- Minor video time effects can be created by setting INPUT_FPS and OUTPUT_FPS to different relative values<br/>
- INPUT_FPS > OUTPUT_FPS creates a slowed time effect
- INPUT_FPS = OUTPUT_FPS creates no time effect
- INPUT_FPS < OUTPUT_FPS creates a timelapse effect -->


## Usage
```
$ conda create --name env
$ conda activate env
$ conda install -c conda-forge opencv
```
Before proceding, if CUDA is available on your machine, now follow this <a href="https://www.tensorflow.org/install/pip?hl=it">guide</a> to make Tensorflow expoit your GPU. 
```
$ pip install -r requirements.txt
$ python style_frames.py -i *input_video_path* [-args]
```

## Examples
### Input Video
![file](/examples/reference.gif)
### Example 1
##### Reference Style Image Transition Sequence
![file](/examples/example1_style_sequence.png)
##### Output Video
![file](/examples/example1.gif)
##### Output Video with Preserved Colors
![file](/examples/example3.gif)
### Example 2
##### Reference Style Image Transition Sequence
![file](/examples/example2_style_sequence.png)
##### Output Video
![file](/examples/example2.gif)

##### [Example Video made using this program](https://youtu.be/vgl83UTciD8) 
