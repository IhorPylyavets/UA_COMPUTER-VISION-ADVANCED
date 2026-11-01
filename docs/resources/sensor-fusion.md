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

## Датасети та бенчмарки

**RGB + LiDAR / Radar:**

| Датасет | Сенсори | Для чого |
|---|---|---|
| [KITTI 3D Object](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) | стерео + LiDAR | Класичний старт для 3D-детекції |
| [nuScenes](https://www.nuscenes.org/) | 6 камер, LiDAR, 5 радарів, 360° | Головний датасет із радаром; NDS / mAP |
| [Waymo Open](https://waymo.com/open/), [Argoverse 2](https://www.argoverse.org/av2.html) | камери + кілька LiDAR, великий масштаб | Масштабні 3D-детекція і трекінг |
| [View-of-Delft](https://github.com/tudelft-iv/view-of-delft-dataset) | камера + LiDAR + 4D-радар | Злиття з 4D-радаром |
| [K-Radar](https://github.com/kaist-avelab/K-Radar) | 4D-радар (тензори), камери, LiDAR; сніг, дощ, туман | Радар у погоду, де камера сліпне |

**RGB + тепловізор:**

| Датасет | Що в ньому | Для чого |
|---|---|---|
| [FLIR ADAS v2](https://adas-dataset-v2.flirconservator.com/) | пари RGB + LWIR з авто, детекційна розмітка | Детекція в IR |
| [DroneVehicle](https://github.com/VisDrone/DroneVehicle) | пари RGB + IR **з дрона**, повернуті bbox, 5 класів техніки | Найближче до треку 3 фінального проєкту |
| [VEDAI](https://downloads.greyc.fr/vedai/) | аерознімки RGB + ближній IR (не тепловізор), техніка | Малий класичний бенчмарк |
| [LasHeR](https://github.com/BUGPLEASEOUT/LasHeR), [VTUAV](https://zhang-pengyu.github.io/DUT-VTUAV/) | RGB-T трекінг; VTUAV — з дрона | RGB-T SOT |
| [SIRST](https://github.com/YimianDai/sirst), [NUDT-MIRSDT](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM) | малі тьмяні цілі в IR; MIRSDT — багатокадровий | Track-before-detect (слайд 31 заняття 9) |
