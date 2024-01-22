### <p align="center">FocusFlow: Boosting Key-Points Optical Flow Estimation for Autonomous Driving
<br>
<div align="center">
  <b>
  <a href="https://www.researchgate.net/profile/Zhonghua-Yi" target="_blank">Zhonghua&nbsp;Yi</a> &middot;
  <a href="https://www.researchgate.net/profile/Shi-Hao-10" target="_blank">Hao&nbsp;Shi</a> &middot;
  <a href="https://www.researchgate.net/profile/Kailun-Yang" target="_blank">Kailun&nbsp;Yang</a> &middot;
  <a href="https://www.researchgate.net/profile/Qi-Jiang-63" target="_blank">Qi&nbsp;Jiang</a> &middot;
  <a href="https://www.researchgate.net/profile/Yaozu-Ye" target="_blank">Yaozu&nbsp;Ye</a> &middot;
  <a href="https://www.researchgate.net/profile/Ze-Wang-42" target="_blank">Ze&nbsp;Wang</a> &middot;
  <a href="https://www.researchgate.net/profile/Kaiwei-Wang-4" target="_blank">Kaiwei&nbsp;Wang</a> 
  </b>
  <br> <br>

  <a href="https://ieeexplore.ieee.org/abstract/document/10258386" target="_blank">IEEE Transactions on Intelligent Vehicles</a>
  <br>
  <a href="https://arxiv.org/abs/2308.07104" target="_blank">Arxiv</a>
  <br>
  <a href="https://www.researchgate.net/profile/Kailun-Yang/publication/374095317_FocusFlow_Boosting_Key-Points_Optical_Flow_Estimation_for_Autonomous_Driving" target="_blank">ResearchGate</a>
  

####
</div>
<br>
<p align="center">:hammer_and_wrench: :construction_worker: :rocket:</p>
<p align="center">:fire: The keypoints mask data and pretrained models are now available. :fire:</p>
<br>

<div align=center><img src="assets/comparison.png" width="1000" height="361" /></div>

[comment]: <> (### Update)

[comment]: <> (- 2022.11.21 Release the [arXiv]&#40;https://arxiv.org/abs/2211.11293&#41; version with supplementary materials.)

### Update
- 2023.08.14 Init repository.
- 2024.1.10 Code release.
- 2024.1.21 Pretrained models and mask data release.

### TODO List
1. [x] Code release. 
2. [x] Pretrained models release.
3. [x] Mask data release.

### Setup the environment
We recommend using Anaconda to set up the environment.
```bash
conda create -n focusflow python=3.10
conda activate focusflow
pip install -r requirements.txt
```

### Dataset
The following datasets are required for training and testing:
- [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)
- [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
- [Sintel](http://sintel.is.tue.mpg.de/downloads)

Since the official KITTI and Sintel benchmark do not provide optical flow benchmark on keypoints, 
we randomly split the training set of Sintel and KITTI into training and validation sets. The split details are provided in the `Sintel_split.txt` and `KITTI_split.txt` files.

Additionally, we provide the preprocessed keypoints masks for SIFT, ORB, GoodFeature and SiLK in the `data` folder.
The SIFT, ORB and GoodFeature keypoints are extracted using the [OpenCV](https://opencv.org/) library, while the SiLK keypoints are extracted using the [SiLK](https://github.com/facebookresearch/silk) library.
Scripts used for generating keypoint masks are provided in the `scripts` folder.

By default, the dataset is expected to be stored in the following structure:
```
data
├── FlyingChairs_release
│   ├── data
│   ├── FlyingChairs_train_val.txt
├── FlyingThings3D
│   ├── frames_cleanpass
│   ├── frames_finalpass
│   ├── optical_flow
├── KITTI
│   ├── training
│   │   ├── image_2
│   │   ├── flow_occ
│   ├── val
├── Sintel
│   ├── training
│   │   ├── clean
│   │   ├── final
│   │   ├── flow
│   ├── val
├── mask
│   ├── FlyingChairs_release
│   │   ├── orb
│   │   ├── sift
│   │   ├── goodfeature
│   │   ├── silk
│   ├── FlyingThings3D
│   ├── KITTI
│   ├── Sintel
```
The `mask` data could be downloaded in [OneDrive](https://1drv.ms/f/s!AlFuh7JTFAc3i-wcIh6mr64hlXm7TA?e=VBIoN5).

### Usage
To use the specific model, please run the training or evaluation script in the `core/models/{model_name}` folder.

For example, to train the FocusRAFT model for ORB points, please run the following command:
```bash
cd core/models/ff-raft
python train.py --yaml configs/experiment/ffraft_chairs_orb.yaml
```

The pretrained model are supposed to be stored in the `pretrain` folder in each model's folder.
Pretrained models could be downloaded in [OneDrive](https://1drv.ms/f/s!AlFuh7JTFAc3i-wcIh6mr64hlXm7TA?e=VBIoN5).

### Abstract
Key-point-based scene understanding is fundamental for autonomous driving applications. 
At the same time, optical flow plays an important role in many vision tasks. 
However, due to the implicit bias of equal attention on all points, classic data-driven optical flow estimation methods yield less satisfactory performance on key points, limiting their implementations in key-point-critical safety-relevant scenarios. 
To address these issues, we introduce a points-based modeling method that requires the model to learn key-point-related priors explicitly. Based on the modeling method, we present FocusFlow, a framework consisting of 1) a mix loss function combined with a classic photometric loss function and our proposed Conditional Point Control Loss (CPCL) function for diverse point-wise supervision; 2) a conditioned controlling model which substitutes the conventional feature encoder by our proposed Condition Control Encoder (CCE). 
CCE incorporates a Frame Feature Encoder (FFE) that extracts features from frames, a Condition Feature Encoder (CFE) that learns to control the feature extraction behavior of FFE from input masks containing information of key points, and fusion modules that transfer the controlling information between FFE and CFE. 
Our FocusFlow framework shows outstanding performance with up to ${+}44.5\%$ precision improvement on various key points such as ORB, SIFT, and even learning-based SiLK, along with exceptional scalability for most existing data-driven optical flow methods like PWC-Net, RAFT, and FlowFormer. 
Notably, FocusFlow yields competitive or superior performances rivaling the original models on the whole frame.

## Method

<p align="center">
    Conditional Point Control Loss (CPCL)
</p>
<p align="center">
    <div align=center><img src="assets/CPCL.png" width="777" height="530" /></div>
<br><br>

<p align="center">
    Conditional Architecture
</p>
<p align="center">
    <div align=center><img src="assets/conditional_architecture.png" width="776" height="409" /></div>
<br><br>

<p align="center">
    The FocusFlow Framework
</p>
<p align="center">
    <div align=center><img src="assets/framework.png" width="1000" height="415" /></div>
<br><br>

### Reference and License
The code is based on the following open-source project:

- [RAFT](https://github.com/princeton-vl/RAFT) (BSD 3-Clause License)
- [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official) (Apache-2.0 License)
- [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc) (GPL-3.0 License)

Due to the use of the above open-source projects, our code is under the GPL-3.0 License.

### Citation
```
[1]  @article{yi2023focusflow,
        title={FocusFlow: Boosting Key-Points Optical Flow Estimation for Autonomous Driving},
        journal={IEEE Transactions on Intelligent Vehicles},
        year={2023},
        publisher={IEEE}
    }
```

### Contact
Feel free to contact me if you have additional questions or have interests in collaboration. Please drop me an email at yizhonghua@zju.edu.cn. =)