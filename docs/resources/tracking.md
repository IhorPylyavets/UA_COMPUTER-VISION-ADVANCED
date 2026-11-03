# Трекінг: збірки методів на GitHub

Заняття 9–10, трек 4 фінального проєкту. Тут лише репозиторії, де під одним API зібрано багато методів, а не один трекер.
Стан — листопад 2026: перед використанням перевір дату останнього коміту.

## SOT — один об'єкт

| Репозиторій | Методи | Примітка |
|---|---|---|
| [visionml/pytracking](https://github.com/visionml/pytracking) | ATOM, DiMP, PrDiMP, SuperDiMP, KYS, KeepTrack, ToMP, LWL, RTS | Основний вибір: спільна інфраструктура, навчання (`ltr`), оцінка на LaSOT / GOT-10k / TrackingNet |
| [STVIR/pysot](https://github.com/STVIR/pysot) | SiamRPN, DaSiamRPN, SiamRPN++, SiamMask | Сімейство сіамських трекерів (SenseTime) |
| [HonglinChu/SiamTrackers](https://github.com/HonglinChu/SiamTrackers) | SiamFC, SiamRPN, DaSiamRPN, SiamRPN++, SiamMask, SiamFC++, SiamCAR, NanoTrack | Сіамські трекери в одному стилі, є легкі для edge |
| [open-mmlab/mmtracking](https://github.com/open-mmlab/mmtracking) | SiamRPN++, STARK, PrDiMP, MixFormer + MOT і VID | Не розвивається з 2023; MOT переїхав у MMDetection 3.x |
| OpenCV (`cv2.Tracker*`) | KCF, CSRT, MIL, DaSiamRPN, NanoTrack, ViT | Класика без навчання, для швидких експериментів |

## MOT — багато об'єктів, одна камера

| Репозиторій | Методи | Примітка |
|---|---|---|
| [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot) | ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT, HybridSORT, BoostTrack, SFSORT | Найповніша Python-бібліотека; будь-який детектор і ReID |
| [roboflow/trackers](https://github.com/roboflow/trackers) | SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU, McByte | Чистий модульний код, Apache 2.0 |
| [JackWoo0831/Yolov7-tracker](https://github.com/JackWoo0831/Yolov7-tracker) | SORT, DeepSORT, ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT, C-BIoU, SparseTrack, UCMCTrack, HybridSORT | Один стиль коду, детектор / трекер / ReID / Калман розв'язані — зручно для абляцій |
| [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (`configs/mot`) | JDE, FairMOT, MCFairMOT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT, CenterTrack | Є й MTMC (див. [MTMC](#mtmc--багато-камер)) |
| [Geekgineer/motcpp](https://github.com/Geekgineer/motcpp) | SORT, ByteTrack, OC-SORT, BoT-SORT, StrongSORT, Deep OC-SORT, BoostTrack | C++ для real-time |

## Метрики

[TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, MOTA, IDF1) — уже в кореневому `requirements.txt`.

## MTMC — багато камер

Бібліотек із набором MTMC-методів під одним API фактично немає: є повні конвеєри-baseline
і окремі рішення AI City Challenge.

| Репозиторій | Що всередині |
|---|---|
| [PaddleDetection, PP-Tracking](https://github.com/PaddlePaddle/PaddleDetection) | MOT → ReID → міжкамерна асоціація (MTMCT); офіційно baseline без оптимізацій під топологію сцени |
| [regob/vehicle_mtmc](https://github.com/regob/vehicle_mtmc) | YOLOv5 + DeepSORT / ByteTrack, міжкамерне зіставлення ієрархічною кластеризацією, MOTA / IDF1 |
| [LCFractal/AIC21-MTMC](https://github.com/LCFractal/AIC21-MTMC) | 1-ше місце AI City 2021 (Track 3), зони перехресть |
| [Yejin0111/AICITY2022-Track1-MTMC](https://github.com/Yejin0111/AICITY2022-Track1-MTMC) | 1-ше місце AI City 2022 (Track 1), на PaddleDetection |
| [coder-wangzhen/AIC22-MCVT](https://github.com/coder-wangzhen/AIC22-MCVT) | 2-ге місце AI City 2022 (Track 1) |
| [SherryJYC/paper-MTMC](https://github.com/SherryJYC/paper-MTMC) | Список статей з MTMC |

## Датасети та бенчмарки

**MOT:**

| Датасет | Що в ньому | Для чого |
|---|---|---|
| [MOT17 / MOT20](https://motchallenge.net/data/MOT17/) | пішоходи, статичні й рухомі камери; MOT20 — дуже щільні натовпи | Стандартна таблиця HOTA / IDF1 (слайд 22) |
| [DanceTrack](https://dancetrack.github.io/), [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | однаковий вигляд, нелінійний рух | Асоціація, коли ReID не допомагає |
| [VisDrone-MOT](https://github.com/VisDrone/VisDrone-Dataset), [UAVDT](https://sites.google.com/view/grli-uavdt/) | траєкторії з дронів, дрібні об'єкти, рух камери | Найближче до розвідувального БпЛА (кейс 1) |
| [MDMT](https://huggingface.co/datasets/mrhwl123/MDMT) | два дрони, спільні ID між ними (CVAT XML) | Демо заняття 9 і ДЗ 4 |

**SOT:**

| Датасет | Що в ньому | Для чого |
|---|---|---|
| [LaSOT](http://vision.cs.stonybrook.edu/~lasot/), [GOT-10k](http://got-10k.aitestunion.com/), [TrackingNet](https://github.com/SilvioGiancola/TrackingNet-devkit) | довгі послідовності; GOT-10k — класи тесту не перетинаються з train | Стандартна трійка для SOT |
| [UAV123](https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset), [DTB70](https://github.com/flyers/drone-tracking) | супровід з дрона, швидкий рух камери | SOT на гімбалі |
| [Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) | RGB + IR відео з дронами-цілями | Супровід дронів, IR (кейс 2) |
| [VOT Challenge](https://www.votchallenge.net/) | щорічний челендж: short-term, long-term, RGB-T, RGB-D | Перевірка на свіжих даних |

**MTMC і ReID:**

| Датасет | Що в ньому | Для чого |
|---|---|---|
| [AI City Challenge](https://www.aicitychallenge.org/) (CityFlowV2) | міська мережа камер, авто; щорічні треки MTMC | Основний бенчмарк MTMC для техніки |
| [WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) | 7 камер з перекриттям, каліброваних, пішоходи | Асоціація через спільну площину землі |
| [MMPTrack](https://iccv2021-mmp.github.io/), [MTMMC](https://sites.google.com/view/mtmmc) | MMPTrack — багатокамерні сцени в приміщенні; MTMMC — кампус і цех, RGB + тепловізор | Довгі ID через багато камер |
| [VeRi-776](https://github.com/VehicleReId/VeRidataset) | ReID авто з 20 камер | ReID техніки |
| [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html), [MSMT17](http://www.pkuvmc.com/dataset.html) | ReID пішоходів | Стандартні бенчмарки (зручно через [torchreid](https://github.com/KaiyangZhou/deep-person-reid)). DukeMTMC автори відкликали — не використовуйте |

Ліцензії майже всіх перелічених датасетів — лише для досліджень; перед використанням у продукті перевір.
