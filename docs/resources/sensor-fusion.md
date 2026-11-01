# Злиття сенсорів: збірки методів на GitHub

Заняття 11–12, трек 3 фінального проєкту. Тут лише репозиторії, де під одним API зібрано багато методів.
Стан — листопад 2026: перед використанням перевір дату останнього коміту.

## RGB + LiDAR / Radar (3D-детекція)

| Репозиторій | Методи | Примітка |
|---|---|---|
| [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | PointPillars, SECOND, PointRCNN, Part-A², PV-RCNN / PV-RCNN++, Voxel R-CNN, CenterPoint, TransFusion, BEVFusion, DSVT, VoxelNeXt, MPPNet | Основний вибір для LiDAR і LiDAR+камера; KITTI, nuScenes, Waymo |
| [open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d) | PointPillars, SECOND, CenterPoint, PV-RCNN, MVXNet, BEVFusion, FCOS3D, PGD, SMOKE, ImVoxelNet, ImVoteNet, DSVT | Ширше за модальностями (є моно-3D і RGB+LiDAR), але не розвивається з 2024 |
| [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) | BEVFusion: камера + LiDAR у спільному BEV | Архівовано; ті самі моделі є в OpenPCDet і MMDetection3D |
| [nutonomy/nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) | завантаження nuScenes (6 камер, LiDAR, 5 радарів), проєкції, метрики NDS / mAP | Головний датасет із радаром |

## RGB + тепловізор

| Репозиторій | Що всередині | Примітка |
|---|---|---|
| [Linfeng-Tang/VIF-Benchmark](https://github.com/Linfeng-Tang/VIF-Benchmark) | понад 15 методів злиття visible + IR: DenseFuse, NestFuse, RFN-Nest, U2Fusion, FusionGAN, GANMcC, SeAFusion, SwinFusion, TarDAL, PIAFusion, SuperFusion та ін. + спільні метрики | Єдина збірка методів злиття зображень під одним запуском; MIT |
| [bupt-ai-cz/LLVIP](https://github.com/bupt-ai-cz/LLVIP) | датасет: вирівняні пари RGB + LWIR вночі, пішоходи | Найчастіший датасет для early-fusion детекторів |
| [JinyuanLiu-CV/TarDAL](https://github.com/JinyuanLiu-CV/TarDAL) | метод TarDAL і датасет M3FD (RGB + IR, 6 класів, детекційна розмітка) | Цінний передусім датасетом; GPL-3.0 |
| [SoonminHwang/rgbt-ped-detection](https://github.com/SoonminHwang/rgbt-ped-detection) | датасет KAIST Multispectral Pedestrian | Класичний бенчмарк RGB-T детекції пішоходів |

Бібліотеки з багатьма методами RGB-T *детекції* під одним API поки немає: кожна стаття — окремий репозиторій.
