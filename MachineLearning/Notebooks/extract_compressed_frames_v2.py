"""
Modified Frame Extraction Script for DeepFake Detection
Version: 2.0
Date: May 2024

Description:
This script extracts frames from compressed videos, based on the original script provided by Andreas Roessler for the FaceForensics++ dataset.
The script has been modified to include additional functionality, such as specifying the number of frames to extract and additional datasets.

Original Script:
The original script can be found at https://github.com/ondyari/FaceForensics/blob/master/dataset/extract_compressed_videos.py

Original Author:
Andreas Roessler
Date: 25.01.2019

Citation:
If you use this script or the FaceForensics++ dataset, please cite the following paper:
@inproceedings{roessler2019faceforensicspp,
    author = {Andreas R\"ossler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
    title = {Face{F}orensics++: Learning to Detect Manipulated Facial Images},
    booktitle= {International Conference on Computer Vision (ICCV)},
    year = {2019}
}

Modifications:
- Added functionality to specify the number of frames to extract from each video.
- Included additional datasets (original_youtube, DeepFakeDetection, DeepFakeDetection_original).
- Adjusted frame extraction logic to distribute frames evenly.
- Updated the argument parser for the new functionalities.

Author of Modifications: Roman Barron
"""

import os
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm

#Assumes application is ran in directory D:\DataSet\NewDataset
DATASET_PATHS = {
    'original': 'original_sequences',
    'original_youtube': 'original_sequences/youtube',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'DeepFakeDetection_original': 'original_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap'
}
COMPRESSION = ['c0', 'c23', 'c40']


def extract_frames(data_path, output_path, num_frames, method='cv2'):
    os.makedirs(output_path, exist_ok=True)

    if method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure at least one frame is extracted
        if num_frames >= frame_count:
            interval = 1
        else:
            # Calculate the interval to distribute frames evenly
            # We subtract one from num_frames to include both the first and last frames
            interval = frame_count // (num_frames - 1)

        extracted_frames = 0
        for frame_num in range(0, frame_count, interval):
            # Check if we have extracted the desired number of frames
            if extracted_frames < num_frames:
                reader.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                success, image = reader.read()
                if not success:
                    break
                cv2.imwrite(join(output_path, '{:04d}.png'.format(extracted_frames)),
                            image)
                extracted_frames += 1
            else:
                break
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))


def extract_method_videos(dataset, compression, output_path, num_frames=15):

    """Extracts all videos of a specified method and compression in the FaceForensics++ file structure"""
    # Construct the path to the videos directory within the current working directory
    videos_path = join(DATASET_PATHS[dataset], compression, 'videos')

    # Construct the output path for extracted images
    images_path = join(output_path, DATASET_PATHS[dataset], compression, 'images')

    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split('.')[0]
        specific_image_path = join(images_path, image_folder)
        os.makedirs(specific_image_path, exist_ok=True)  # Ensure the directory exists

        # Call the extract_frames function with the specific video path and where to save the extracted frames
        extract_frames(join(videos_path, video), specific_image_path, num_frames=num_frames)





if __name__ == '__main__':
    print("Usage: script.py -d DATASET --compression COMPRESSION_LEVEL --num_frames NUM_FRAMES --output_path PATH")
    print("Where:")
    print("--dataset DATASET specifies the dataset to use. Choices are: " + ', '.join(list(DATASET_PATHS.keys()) + ['all']) + ". 'all' runs for all datasets.")
    print("--compression COMPRESSION_LEVEL specifies the compression level. Choices are: " + ', '.join(COMPRESSION) + ".")
    print("--num_frames NUM_FRAMES specifies the number of frames to extract from each video. Default is 15.")
    print("--output_path NUM_FRAMES specifies the number of frames to extract from each video. Default is 15.")

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', '-d', type=str, choices=list(DATASET_PATHS.keys()) + ['all'], default='all', help="Specify the dataset to use.")
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION, default='c0', help="Specify the compression level.")
    p.add_argument('--num_frames', type=int, default=15, help='Number of frames to extract from each video.')
    p.add_argument('--output_path', type=str, required=True, help="Specify the path where extracted frames should be saved.")

    args = p.parse_args()

    # Directly using args.output_path without prepending a data_path
    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            extract_method_videos(dataset, args.compression, args.output_path, num_frames=args.num_frames)
    else:
        extract_method_videos(args.dataset, args.compression, args.output_path, num_frames=args.num_frames)
