# test the pre-trained model on a single video
# (working on it)
# Bolei Zhou and Alex Andonian
#
# adopted by Alexey Chaykin (alexey.chaykin@gmail.com) for 
# video based evaluation of epic-kitchen pretrained models 

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
import moviepy.editor as mpy
import time

import torchvision
import torch.nn.parallel
import torch.optim
from tsn import TSN
from tsn import TRN
from torchvision.transforms import Compose
from transforms import GroupScale, GroupCenterCrop, GroupOverSample, Stack, ToTorchFormatTensor, GroupNormalize
from torch.nn import functional as F

from model_loader import load_checkpoint, make_model

def extract_frames(video_file, num_frames=8):
    try:
        os.makedirs(os.path.join(os.getcwd(), 'frames'))
    except OSError:
        pass

    output = subprocess.Popen(['ffmpeg', '-i', video_file],
                              stderr=subprocess.PIPE).communicate()
    # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
    re_duration = re.compile('Duration: (.*?)\.')
    duration = re_duration.search(str(output[1])).groups()[0]

    seconds = functools.reduce(lambda x, y: x * 60 + y,
                               map(int, duration.split(':')))
    rate = num_frames / float(seconds)

    output = subprocess.Popen(['ffmpeg', '-i', video_file,
                               '-vf', 'fps={}'.format(rate),
                               '-vframes', str(num_frames),
                               '-loglevel', 'panic',
                               'frames/%d.jpg']).communicate()
    frame_paths = sorted([os.path.join('frames', frame)
                          for frame in os.listdir('frames')])

    frames = load_frames(frame_paths)
    subprocess.call(['rm', '-rf', 'frames'])
    return frames


def load_frames(frame_paths, num_frames=8):
    frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames


# options
parser = argparse.ArgumentParser(description="test Epic-Kitchen TSN/TRN models on a single video")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--video_file', type=str, default=None)
group.add_argument('--frame_folder', type=str, default=None)
parser.add_argument('--rendered_output', type=str, default=None)
parser.add_argument('--weights', type=str)
parser.add_argument('--verb_categories', type=str, default="verb_categories.txt")
parser.add_argument('--noun_categories', type=str, default="noun_categories.txt")
parser.add_argument('--verb_ground_truth', type=str, default=None)
parser.add_argument('--noun_ground_truth', type=str, default=None)

args = parser.parse_args()

verb_categories = [line.rstrip() for line in open(args.verb_categories, 'r').readlines()]
noun_categories = [line.rstrip() for line in open(args.noun_categories, 'r').readlines()]

# Load model.

net = load_checkpoint(args.weights)
net.eval()

# Initialize frame transforms.

crop_count = 1
backbone_arch = net.base_model # "resnet50" 
num_segments = net.num_segments # 8 

if crop_count == 1:
    cropping = Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif crop_count == 10:
    cropping = GroupOverSample(net.input_size, net.scale_size)
else:
    raise ValueError("Only 1 and 10 crop_count are supported while we got {}".format(crop_count))

transform = Compose([
    cropping,
    Stack(roll=backbone_arch == 'BNInception'),
    ToTorchFormatTensor(div=backbone_arch != 'BNInception'),
    GroupNormalize(net.input_mean, net.input_std),
])

# Obtain video frames
if args.frame_folder is not None:
    import glob
    # Here, make sure after sorting the frame paths have the correct temporal order
    frame_paths = sorted(glob.glob(os.path.join(args.frame_folder, '*.jpg')))
    frames = load_frames(frame_paths)
else:
    frames = extract_frames(args.video_file, num_segments)

start = time.time()

# Make video prediction.
data = transform(frames)
input = data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0) #.cuda()

with torch.no_grad():
    features = net.features(input)
    logits = net.logits(features)
  
    verb_h_x = torch.mean(F.softmax(logits[0], 1), dim=0).data
    noun_h_x = torch.mean(F.softmax(logits[1], 1), dim=0).data

    verb_probs, verb_idx = verb_h_x.sort(0, True)
    noun_probs, noun_idx = noun_h_x.sort(0, True)

end = time.time()
time_taken = end-start
 
# Count how far the expected ground truth verb and noun are in the detections lists
max_position = 10
if args.verb_ground_truth is not None and args.noun_ground_truth is not None:
    verb_position = max_position
    noun_position = max_position
    verb_prob = 0.0
    noun_prob = 0.0
    verb_prob_false = 0.0
    noun_prob_false = 0.0
    for i in range(0, max_position):
        if verb_categories[verb_idx[i]]==args.verb_ground_truth:
            verb_position = i
            verb_prob = verb_probs[i]
        else: 
            if verb_prob_false < verb_probs[i]:
                verb_prob_false = verb_probs[i]
        if noun_categories[noun_idx[i]]==args.noun_ground_truth:
            noun_position = i
            noun_prob = noun_probs[i]
        else: 
            if noun_prob_false < verb_probs[i]:
                noun_prob_false = verb_probs[i]
    print("%s %d %f %f %s %d %f %f %f" % (args.verb_ground_truth, verb_position, verb_prob, verb_prob_false, args.noun_ground_truth, noun_position, noun_prob, noun_prob_false, time_taken))
else:
# Just output the prediction.
    video_name = args.frame_folder if args.frame_folder is not None else args.video_file
    print('RESULT ON ' + video_name)
    for i in range(0, 10):
        print('{:.3f} -> {}'.format(verb_probs[i], verb_categories[verb_idx[i]]))
        print('{:.3f} -> {}'.format(noun_probs[i], noun_categories[noun_idx[i]]))

# Render output frames with prediction text.
if args.rendered_output is not None:
    verb_prediction = verb_categories[verb_idx[0]]
    noun_prediction = noun_categories[noun_idx[0]]
    rendered_frames = render_frames(frames, verb_prediction+' '+noun_prediction)
    clip = mpy.ImageSequenceClip(rendered_frames, fps=4)
    clip.write_videofile(args.rendered_output)
