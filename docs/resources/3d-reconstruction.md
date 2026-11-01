# 3D-реконструкція: збірки методів на GitHub

Заняття 4. Тут лише репозиторії, де під одним API зібрано багато методів, а не одна стаття.
Стан — листопад 2026: перед використанням перевір дату останнього коміту.

## SfM: пози камер

| Репозиторій | Що всередині | Примітка |
|---|---|---|
| [colmap/colmap](https://github.com/colmap/colmap) | інкрементний і глобальний SfM, MVS, `pycolmap` | Вхід для всіх NeRF/3DGS. Глобальний mapper — це колишній [GLOMAP](https://github.com/colmap/glomap), який тепер архівовано і влито в COLMAP |
| [cvg/Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) | SfM на навчених фічах: SuperPoint, DISK + SuperGlue, LightGlue, LoFTR | Коли SIFT у COLMAP не знаходить пар (аерозйомка, слабка текстура) |

## NeRF і Gaussian Splatting

| Репозиторій | Методи | Примітка |
|---|---|---|
| [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | nerfacto, instant-ngp, mip-NeRF, vanilla NeRF, TensoRF, NeuS, D-NeRF, splatfacto (3DGS), splatfacto-mcmc | Основний вибір: один CLI для даних, навчання, переглядача й експорту; Apache-2.0. Останній коміт — 2025 |
| [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | растеризатор 3DGS, MCMC-densification, anti-aliasing, 2DGS | CUDA-бекенд splatfacto; Apache-2.0 |
| [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | оригінальний 3DGS | Еталон для порівняння; ліцензія лише некомерційна |
| [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) | hash-grid NeRF, SDF, зображення | Найшвидший старт з GUI; ліцензія NVIDIA (некомерційна) |

Список статей і коду з 3DGS: [MrNeRF/awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting).

## Датасети та бенчмарки

| Датасет | Що в ньому | Для чого |
|---|---|---|
| [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) | 9 необмежених сцен 360° | Стандартна таблиця PSNR / SSIM / LPIPS для NeRF і 3DGS |
| [Tanks and Temples](https://www.tanksandtemples.org/) | великі сцени з лазерним GT, онлайн-оцінка | Реконструкція геометрії, а не лише рендер |
| [DL3DV-10K](https://dl3dv-10k.github.io/DL3DV-10K/) | понад 10 тис. відео сцен з позами | Масштабна оцінка й навчання feed-forward моделей |
| [Mill 19](https://github.com/cmusatyalab/mega-nerf), [UrbanScene3D](https://vcc.tech/UrbanScene3D) | аерозйомка великих територій з дрона | Найближче до задач курсу: NeRF / 3DGS на аерофото |
