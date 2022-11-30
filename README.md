# FSErasing: Improving Face Recognition with Data Augmentation Using Face Parsing

## Introduction
We proposes a data augmentation method, called *Face Semantic Erasing (FSErasing)*, for face recognition using face parsing.
Face recognition models are trained with face images erased random face semantic regions such as hair, cheek, forehead, nose, and eye.
We also propose the original face semantic labels with 25 classes, which include 9 additional classes: ``right_cheek``, ``left_cheek``, ``right_chin``, ``left_chin``, ``right_forehead``, ``left_forehead``, ``middle_forehead``, ``around_right_eye``, ``around_left_eye``.

This repository contains the following used for the results in [our paper]():
- implementation of FSErasing
- implementation of the visualization method for face recognition models using face parsing, which called *Face Semantic Class Activation Mapping (FS-CAM)* in this repositoty
- our original semantic labels with 25 classes for detailed face parsing

## Requirements
- Python 3.x (recommended >= 3.8.8)
- pytorch (recommemded >= 1.8.1)
- pandas (recommended >= 1.2.4)
- opencv-python (recommended >= 4.5.1.48) 

## Dataset
### Downloading the dataset
You can download the detailed face semantic labels (with 25 classes) for FaceSynthetics dataset [^1] from the link below.

[Google Drive]()


Note that the face images and landmark labels are **NOT** included in our distributed files.
They are available for download at [the official repository of FaceSynthetics](https://github.com/microsoft/FaceSynthetics) (full dataset of 100,000 images).

### Dataset layout
The detailed face semantic labels are contained in a single .zip file.

```
detailed_facesynthetics.zip
├── labels25
|   └── {frame_id}_seg.png    # Segmentation image, where each pixel has an integer value mapping to the categories below
|
```

## Pre-trained face parsing models

## Example notebooks

## Acknowledgments


[^1]: E. Wood, T. Baltrusaitis, C. Hewitt, S. Dziadzio, T.J. Cashman, and J. Shotton, "Fake It Till You Make It: Face analysis in the wild using synthetic data alone," Proc. Int'l Conf. Computer Vision (ICCV), pp. 3681--3691, Oct. 2021.
