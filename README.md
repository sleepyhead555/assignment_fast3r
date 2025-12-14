# assignment_fast3r
本项目为深度学习选修课大作业，使用了Fast3R(CVPR 2025)的开源代码，目的是通过fast3r开源的代码，编写代码实现生成点云图、估计相机位姿并与真值进行比较等功能

虽然官方给的要求是cuda=12.4，但是cuda=11.8也足以完成我所用到的功能
本次尝试并未重新进行训练，使用的是官方的预训练权重Fast3R_ViT_Large_512

首先，用Fast3R官方提供的视频试手，看一下效果。编写代码run_from_video.py，选择demo_examples/kitchen/family/Family.mp4这个视频，将其抽帧输入，实现获取模型估计的相机位姿、生成点云等功能，在这一过程中发现并解决了一些官方文件中存在的一些小纰漏，详情见文件“我做过的改动”。

之后，编写7scenes_test.py代码（设计了三种读取照片的方式strategy，即linspace,stride,window），通过7scene数据集图像实现我的估计相机位姿、生成点云与评估功能。评估指标包括，点云重建评估指标：accuracy（Acc，准确度），completion（Comp，完整度），normal consistency（NC，法线一致性），completion ratio（完整度比率）；相机位姿评估指标Area Under Curve at 5 degrees（ACU@5，5度阈值下的曲线下面积），Absolute Translation Error（ATE，绝对平移误差），Absolute Rotation Error（绝对旋转误差），Relative Pose Error - Rotation（RPE-rot，相对位姿误差-旋转），Relative Pose Error - Translation（RPE-trans，相对位姿误差-平移）

代码面向其中一个scene的一个seq的图片，通过其中五张图进行点云重建与相机位姿估计，以及重建效果的评估。我现在提供了7scenes_test.py根据chess/seq-01的五张图生成的点云图效果
