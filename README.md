# SurgRIPE Challenge Submission

An improvement of PVNet for participating in the Surgical Robot Instrument Pose Estimation (SurgRIPE) challenge, which is a part of the Structured description of the challenge design of the Endoscopic Vision Challenge during MICCAI 23 in Vancouver.

This work is base on PVNet.

> [PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation](https://arxiv.org/pdf/1812.11788.pdf)  
> Sida Peng, Yuan Liu, Qixing Huang, Xiaowei Zhou, Hujun Bao   
> CVPR 2019 oral  
> [Project Page](https://zju3dv.github.io/pvnet)

Any questions or discussions are welcomed!

## Improvements over Original PVNet
- **Data Augmentation**: Additional data augmentation were employed, which increases the robustness during occlusions.
- **Hyperparameters & Settings**: Modifications were made to certain hyperparameters and settings, optimizing them for the challenge.


## Installation

Set up the environment with docker. See [this](https://github.com/Acmonds/clean-pvnet/tree/master/docker).



## Testing

Run the command below, and the script will automatically download the preprocessing files and weight files.

1. Prepare the data:
    ```
    python run.py --func preprocess --path /home/clean-pvnet/Dataset/[MBF/LND]/[TEST/TEST_OCC] --type [m/l]
    ```
2. Test:
    ```
    python run.py --func evaluate --path /home/clean-pvnet/Dataset/[MBF/LND]/[TEST/TEST_OCC] --type [m/l]
    ```
    


## Data structure:

Organize the dataset as the following structure:
```
    ├── /path/to/dataset
    │   ├── model.ply
    │   ├── camera.txt
    │   ├── diameter.txt  // the object diameter, whose unit is meter
    │   ├── rgb/
    │   │   ├── 0.jpg
    │   │   ├── ...
    │   │   ├── 1234.jpg
    │   │   ├── ...
    │   ├── mask/
    │   │   ├── 0.png
    │   │   ├── ...
    │   │   ├── 1234.png
    │   │   ├── ...
    │   ├── pose/
    │   │   ├── pose0.npy
    │   │   ├── ...
    │   │   ├── pose1234.npy
    │   │   ├── ...
    │   │   └──
```
