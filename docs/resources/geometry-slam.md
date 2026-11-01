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
