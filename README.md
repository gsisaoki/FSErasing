# FSErasing: Improving Face Recognition with Data Augmentation Using Face Parsing

## Introduction
We proposes a data augmentation method, called Face Semantic Erasing (FSErasing), for face recognition using face parsing.
Face recognition models are trained with face images erased random face semantic regions such as hair, cheek, forehead, nose, and eye.
We also propose the original face semantic labels with 25 classes, which include 9 additional classes: ``right_cheek``, ``left_cheek``, ``right_chin``, ``left_chin``, ``right_forehead``, ``left_forehead``, ``middle_forehead``, ``around_right_eye``, ``around_left_eye``.

This repository contains the following used for the results in [our paper]():
- implementation of FSErasing
- implementation of the visualization method for face recognition models using face parsing (called FS-CAM in this repositoty)
- our original semantic labels for detailed face parsing

## Requirements and Installation
- pytorch (recommemded >= 1.8.1)
- pandas (recommended >= 1.2.4)
- opencv-python (recommended >= 4.5.1.48) 

## Downloading the dataset

## Downloading pre-trained face parsing models

## Example notebooks

## Acknowledgement
