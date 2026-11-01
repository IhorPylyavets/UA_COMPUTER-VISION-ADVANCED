# Геометрія, одометрія та SLAM: збірки методів на GitHub

Заняття 2–3. Тут лише репозиторії, де під одним API зібрано багато методів або режимів, а не одна стаття.
Стан — листопад 2026: перед використанням перевір дату останнього коміту.

## Калібрування

| Репозиторій | Що всередині | Примітка |
|---|---|---|
| [opencv/opencv](https://github.com/opencv/opencv) (`calib3d`) | pinhole і fisheye, стерео, ректифікація, `solvePnP`, `StereoSGBM` | База заняття 2 |
| [ethz-asl/kalibr](https://github.com/ethz-asl/kalibr) | multi-camera, camera–IMU, rolling shutter, кілька моделей дисторсії | Стандарт для VIO-платформ; ROS, останній коміт 2024 |

## Особливі точки та зіставлення

| Репозиторій | Методи | Примітка |
|---|---|---|
| [kornia/kornia](https://github.com/kornia/kornia) | SIFT, HardNet, DISK, ALIKED, XFeat, DeDoDe, LoFTR, LightGlue + епіполярна геометрія, RANSAC | Диференційовна геометрія на PyTorch, все на GPU |
| [cvg/Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) | Екстрактори SuperPoint, DISK, D2-Net, SIFT, R2D2; матчери SuperGlue, LightGlue, LoFTR; пошук NetVLAD, OpenIBL, MegaLoc | Конвеєр SfM і візуальної локалізації поверх COLMAP |

## Одометрія та SLAM

| Репозиторій | Що всередині | Примітка |
|---|---|---|
| [luigifreda/pyslam](https://github.com/luigifreda/pyslam) | VO/SLAM з десятками локальних фіч і глобальних дескрипторів, loop closure, depth prediction (DepthAnything, RAFT-Stereo), DUSt3R / MASt3R / VGGT, TSDF і 3DGS | Найкращий «конструктор» для експериментів; GPL-3.0 |
| [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | моно, стерео, RGB-D, кожен з IMU або без, multi-map | Еталонний feature-based SLAM; GPL-3.0, останній коміт 2024 |
| [rpng/open_vins](https://github.com/rpng/open_vins) | MSCKF-VIO, кілька камер, онлайн-калібрування камера–IMU | GPL-3.0 |
| [PRBonn/kiss-icp](https://github.com/PRBonn/kiss-icp) | LiDAR-одометрія на ICP, мінімум параметрів | Лідарний еталон у demo3; MIT |
| [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO) | LiDAR–IMU одометрія (iterated EKF) | GPL-2.0 |
| [TixiaoShan/LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) | LiDAR–IMU–GPS на factor graph | BSD-3-Clause |
| [gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2) | код до підручника «14 лекцій з Visual SLAM» | Для навчання: від Лі-груп до bundle adjustment |

## Метрики

[MichaelGrupp/evo](https://github.com/MichaelGrupp/evo) — ATE і RPE для форматів KITTI, TUM, EuRoC і ROS bag. GPL-3.0.

## Датасети та бенчмарки

| Датасет | Що в ньому | Для чого |
|---|---|---|
| [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | стерео + LiDAR з авто, 22 послідовності, лідерборд | VO / LiDAR-одометрія, demo заняття 3; CC BY-NC-SA |
| [EuRoC MAV](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f) | стерео + IMU з дрона в приміщенні, GT з Vicon / лазертрекера | VIO на дроні |
| [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset), [TUM VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset) | RGB-D і fisheye-стерео + IMU; метрики ATE / RPE звідси | RGB-D SLAM, VIO |
| [UZH-FPV Drone Racing](https://fpv.ifi.uzh.ch/) | агресивні FPV-прольоти: камери, IMU, точний GT | VIO на швидкому маневрі |
| [Hilti SLAM Challenge](https://hilti-challenge.com/) | мультисенсорні послідовності з будмайданчиків, щорічний челендж | SLAM у важких умовах |
| [Middlebury Stereo](https://vision.middlebury.edu/stereo/), [ETH3D](https://www.eth3d.net/) | стереопари й багаторакурсні сцени з точною глибиною, лідерборди | Стерео, demo заняття 2; ETH3D — також SLAM і MVS |
| [HPatches](https://github.com/hpatches/hpatches-dataset), [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/) | зміни ракурсу й освітлення; пари з відомою геометрією | Оцінка й навчання детекторів та матчерів точок |
| [Image Matching Challenge](https://image-matching-challenge.github.io/) | щорічний челендж CVPR на Kaggle: SfM з «диких» фото | Порівняння конвеєрів зіставлення наживо |
