# Заняття 12 · Демо та ноутбуки

Блок 4 · Multi-Sensor Fusion. Мультиспектральне злиття RGB + LWIR: фізика тепловізора (Планк, ε і відбиття неба, crossover),
сигнальний ланцюг мікроболометра (14-біт raw, AGC, FFC), реєстрація двох камер (H∞, паралакс, підігріта мішень, mutual
information, синхронізація) і злиття всередині мережі (early 4-ch з хірургією першого conv, mid two-stream, late / WBF,
modality dropout, аугментація зсувом T). Продовжує заняття 11: та сама конвенція `T_to_from`, бюджет `f·ω·Δt`, валідація на ближніх і далеких.

## Дані

Fusion-моделі навчаються й оцінюються на **реальних вирівняних парах RGB + LWIR**; фізику і геометрію, де потрібна точно відома
відповідь (температура поверхні, `T_rgb_t`, залишок реєстрації в кожному пікселі), показуємо на синтетиці й одразу перевіряємо
на реальних кадрах. Завантаження з кореня репозиторію (≈ 1,3 ГБ, лише потрібні кадри з офіційних архівів через HTTP Range):

```bash
python tools/fetch_rgbt.py                    # LLVIP + FLIR-aligned, підмножини для демо → artifacts/rgbt/
python tools/fetch_flir_adas.py --raw16      # опційно: 60 сирих 16-біт TIFF FLIR ADAS v2 для Demo 1 (≈ 19 МБ, без форми)
python tools/fetch_rgbt.py --raw16 <zip|тека> # або 16-біт TIFF з уже завантаженого zip / теки (FLIR ADAS v2 чи власні)
python tools/fetch_rgbt.py --info             # KAIST, M3FD, MS2, DroneVehicle, DVTOD — де брати вручну
python tools/fetch_llvip.py --list            # лише LLVIP: сцени й розміри; --full, --split, --scenes, --step, --yolo, --raw — див. --help
```

| Датасет | Що це | Як ділимо | У демо |
|---|---|---|---|
| LLVIP (Jia et al., ICCVW 2021) | 1280×1024, пари вирівняні апаратно, 24 нічні + 2 денні сцени, person | спліт авторів за сценами: train 01–18, 25; test 19–24, 26; день — сцени 25 і 26 | кожен 6-й кадр train (2005), кожен 4-й test (866) |
| FLIR-aligned (Zhang et al., ICIP 2020) | 640×512, пари FLIR ADAS v1, вирівняні вручну, день і ніч | спліт авторів `align_train` / `align_validation` (різні поїздки); день / ніч — за яскравістю неба, поріг 128 | кожен 2-й кадр train (2065), увесь validation (1013) |

Ліцензії: LLVIP — лише некомерційне використання; FLIR — умови Teledyne FLIR. Без даних Demo 3–4 переходять на процедурну
синтетику 64×64 (`fuse12.make_dataset`), а реальні розділи Demo 1–2 друкують, як завантажити дані.

## Файли

- `rgbt12.py` — LLVIP і FLIR-aligned: пари, розмітка person, умови день / ніч, масиви для мережі (160×128, люди < 6 px — зона ігнору),
  контраст людини в T і RGB, пошук зсуву T (MI / NCC / градієнти), 16-біт TIFF.
- `fuse12.py` — CenterNet-подібна heatmap-мережа; моделі RGB / T / early / early + dropout / early + зсув / mid, WBF для точок,
  AP з радіусом max(2 px, 0.2·h) і зоною ігнору, bootstrap-CI і **парний bootstrap** для різниць; синтетика як запасний варіант.
- `sim12.py` — синтетична сцена на щоглі 2 м: RGB 1280×720 (HFOV 60°, f ≈ 1108 px) і LWIR 640×512 (HFOV 45°, f ≈ 773 px),
  тепловізор на 5 см праворуч, boresight 0.40° / −0.25° / 0.10°; радіометрія 8–14 мкм, добова модель температур, ядро 14 біт з FPN,
  NETD 50 мК, τ, FFC, картою битих пікселів; калібрувальна мішень з підігрітими колами (нахил поз — параметр `tilt`).
- `panels.py` — інтерактивні панелі (`static=dict(...)` малює без віджетів).

Файли `.py` мають лежати поруч із ноутбуками. Ваги моделей кешуються в `_cache12/` (`real1_*.pt` — реальні дані, `v4_*.pt` — синтетика):
Demo 3 навчає, Demo 4 бере з кешу і дотреновує `early + shift` і `mid`. Без кешу: Demo 3 — ~25 хв, Demo 4 — ~15 хв на 10-ядерному CPU.

| Файл | Мітка на слайдах | Розділ лекції | Що показує |
|---|---|---|---|
| `lesson12_demo1_lwir_signal.ipynb` | LIVE DEMO 1 (слайди «LWIR · Планк і Він», «LWIR · емісивність», «LWIR · умови», «Сенсори · мікроболометр», «Сигнал · AGC», «Сигнал · NN-вхід», «Сигнал · NUC / FFC») | 01–02 Фізика і сигнал | Частки Планка і контраст на 1 K; сцена о 4 / 13 / 21 год і в диму; лист металу, що «показує» небо; crossover у моделі і **реальний контраст людини вдень і вночі** (LLVIP, FLIR-aligned); NETD у counts і шлейф `ω·τ/IFOV`; AGC з гарячим джерелом і **AGC двох реальних датасетів**; пастка `>> 8` (і на реальному 16-біт raw, якщо є); FFC guard біт-у-біт |
| `lesson12_demo2_registration.ipynb` | LIVE DEMO 2 (слайди «Реєстрація · рамки» … «Час · синхронізація») | 03 Реєстрація | Тест «на нескінченності»; залишок `H∞` = `f·B/Z`; гомографія площини; реєстрація з глибиною; калібрування підігрітою мішенню — фронтальні пози проти поз з нахилами на 5 seed; ORB / NCC / MI на синтетиці і **на реальних парах**; **залишкова розбіжність «вирівняних» LLVIP і FLIR-aligned**; `f·ω·Δt` і пари за мітками |
| `lesson12_demo3_early_fusion.ipynb` | LIVE DEMO 3 (слайди «Fusion · early», «Fusion · Ultralytics») | 04 Fusion | Реальний датасет, поріг день / ніч, статистики T двох датасетів; хірургія першого conv (zero / mean / random) зі збереженням функції; fine-tune з RGB-ваг проти з нуля; AP RGB / T / early / early + dropout за умовами; перевірка встановленого Ultralytics: перенос першого conv, 4-сторінковий TIFF з реальної пари, `>> 8`, порядок каналів, HSV / Albumentations |
| `lesson12_demo4_fusion_eval.ipynb` | LIVE DEMO 4 (слайди «Fusion · mid» … «Fusion · оцінка») | 04 Fusion | NMS проти WBF; таблиця умова × модальність з bootstrap-CI і **парний bootstrap** приростів; зсув T −4…+4 px і по діагоналі, early + аугментація зсувом; T = 0 / RGB = 0; параметри, MAC, час |

Дека: `lesson_12_rgb_thermal_90min.pptx` (шаблон заняття 11; 55 основних слайдів + додаток A на 10 слайдів; нотатки лектора — у notes).

## Очікувані числа

Для самоперевірки після Restart & Run All (Demo 3–4 — з тими ж вагами в `_cache12/`; після перенавчання AP може відрізнятися
на ±0.01–0.02). Перевірено на Python 3.14, numpy 2.5, torch 2.15 (CPU), opencv 5.0, ultralytics 8.4.201.

- **Demo 1, синтетика:** пік 300 K — 9.66 мкм; у 8–14 мкм 37.6 %, у 3–5 мкм 1.3 %; +1 K: LWIR 2.63 Вт/м², MWIR 0.21 Вт/м²;
  відносно 1.53 %/K проти 3.63 %/K. О 4:00 лист металу: фізично +5.1 °C, яскравісна −22.1 °C (небо −24.8 °C). Crossover бетонні
  блоки = трава о 6.9 і 17.2 год; людина ≈ розігрітий асфальт з 13.7 до 16.9 год, ΔT о 14:00 = +0.7 K. NETD 50 мК = 7.6 counts,
  1 K = 151 counts; 30 °/с × τ 10 мс = 4.3 px, 90 °/с — 12.8 px. AGC min–max: людина 185 → 110 з гарячим джерелом; перцентиль
  і фіксований — 186 → 186. `>> 8`: 28 рівнів замість 135. FFC: 24 замерзлі кадри з 450, збіг з телеметрією 100 %, стрибок ≈ 0.6 K.
- **Demo 1, реальні дані:** медіана |ΔT| людини (рівні 8-біт T) — FLIR день 9.0, у 44 % людей < 8; FLIR ніч 60.5 (3 %);
  LLVIP ніч 39.0 (12 %); LLVIP день 40.0 (12 %). |ΔRGB| — FLIR день 20.5, FLIR ніч 13.5, LLVIP ніч 12.0. T-канал: LLVIP mean 86.8,
  std 52.9, середнє кадру 47…109; FLIR mean 135.1, std 65.9, середнє кадру 104…163.
  Реальний 14-біт raw FLIR ADAS v2 (`fetch_flir_adas.py --raw16`, 60 кадрів): `>> 8` лишає 14 рівнів, власний мапінг з LO / HI — 230.
- **Demo 2, синтетика:** вісь T на 10 км → (647.24, 364.35) у RGB, boresight 7.7 px. Залишок `H∞`: 7.07 / 3.14 / 2.26 / 1.26 / 0.94 / 0.75 px
  на 8 / 18 / 25 / 45 / 60 / 75 м (= `f·B/Z`); < 1 px з 56.5 м (RGB) і 39.4 м (T). `H(Z0 = 18 м)`: 3.93 / 0 / 1.88 / 2.39 px.
  З глибиною — 0 px; тепловізор бачить 71.6 % RGB-кадру. Калібрування, медіана по 5 seed: фронтально (1.6–3.2 м, ±5°) — RMS 0.28 px,
  |f_t − GT| 63 px, |t_z| 20.7 см, стіна 30.7 px; з нахилами (1.0–2.5 м, ±30°) — RMS 0.07 px, |f_t − GT| 0.4 px, |t_z| 0.1 см,
  стіна 0.78 px. ORB: 0 правильних пар; MI — (+6, −4) = істина; NCC — від'ємна, мінімум (+7, −4) — за 1 px від істини;
  MI поверх `H∞` — (+3, −5). 30 °/с × 16.7 мс = 9.7 px (RGB) / 6.8 px (T).
- **Demo 2, реальні дані:** ORB на парі FLIR-aligned — 43 пари, 0 правильних. Відомий зсув (3, −2) px знайдено в LLVIP / FLIR:
  MI інтенсивностей 43 % / 57 %, NCC max 7 % / 13 %, NCC min 13 % / 13 %, NCC градієнтів 93 % / 50 %, MI градієнтів 100 % / 40 %.
  Залишкова розбіжність (NCC градієнтів, 120 пар): LLVIP p50 4.0 px, p95 5.7 px нативного кадру (0.7 px сітки 160×128);
  FLIR-aligned p50 2.8 px, p95 8.2 px (2.1 px сітки). MI інтенсивностей дає пік на межі перебору в 19 % / 12 % пар.
- **Demo 3:** train 4070 пар, test 1879 (5904 людини + 319 у зоні ігнору); test за умовами: LLVIP ніч 733, LLVIP день 133,
  FLIR ніч 309, FLIR день 704 кадри. zero-init — max |Δвихід| = 0 (mean 4.21, random 5.38); stem YOLOv8n 44.2 → 59.0 M MAC,
  уся модель 8.86 → 8.89 GFLOPs. Fine-tune 1 епоха, AP all: random 0.28, mean 0.27, zero 0.24, з нуля 0.10.
  AP all: RGB 0.17, T 0.37, early 0.42, early + dropout 0.42; T = 0: early 0.02, early + dropout 0.20.
  Ultralytics 8.4.201: «Transferred 355/355», RGB-фільтри скопійовано, 4-й канал випадковий; TIFF 4 × uint16 → uint8, max 63.
- **Demo 4:** AP за умовами (LLVIP ніч / LLVIP день / FLIR ніч / FLIR день / all):

  | модель | LLVIP ніч | LLVIP день | FLIR ніч | FLIR день | all |
  |---|---|---|---|---|---|
  | RGB | 0.26 | 0.36 | 0.12 | 0.21 | 0.17 |
  | T | 0.64 | 0.54 | 0.36 | 0.14 | 0.37 |
  | late WBF | 0.62 | 0.61 | 0.39 | 0.25 | 0.39 |
  | early 4-ch | 0.68 | 0.59 | 0.38 | 0.20 | 0.42 |
  | early + dropout | 0.65 | 0.62 | 0.42 | 0.21 | 0.42 |
  | early + shift | 0.66 | 0.62 | 0.36 | 0.22 | 0.41 |
  | mid two-stream | 0.63 | 0.48 | 0.42 | 0.22 | 0.40 |

  Late NMS all 0.34, late WBF 0.39. Приріст early над кращою одномодальною (парний bootstrap): LLVIP ніч +0.040 [+0.021, +0.060] — є;
  LLVIP день +0.048, FLIR ніч +0.026, FLIR день −0.011 — не доведено. Зсув T по x −4 / −2 / 0 / +2 / +4 px сітки: early
  0.22 / 0.33 / 0.42 / 0.37 / 0.27, early + shift 0.25 / 0.37 / 0.41 / 0.39 / 0.30, late 0.27 / 0.34 / 0.39 / 0.37 / 0.29; RGB-only 0.17.
  T = 0: early 0.02, early + dropout 0.20, mid 0.01; RGB = 0: early 0.27, early + dropout 0.37.

## Домашнє завдання · ДЗ 12

**Реєстрація RGB ↔ LWIR і 4-канальний детектор.** Дані: LLVIP і FLIR-aligned (вирівняні, `python tools/fetch_rgbt.py --full`)
для детектора; FLIR ADAS v2 або DVTOD (невирівняні) — для реєстрації; або власні.

1. Невирівняні пари: ≥ 20 пар, гомографія за ≥ 8 ручними точками або MI / NCC градієнтів; залишок p50 / p95 у px окремо на ближніх
   і далеких об'єктах; overlay ≥ 5 пар. Для FLIR-aligned — виміряйте залишкову розбіжність, як у Demo 2.
2. Loader 4-ch: власний T-мапінг (фіксовані LO / HI з train), тест одного кадру — shape `(H, W, 4)`, dtype, порядок каналів.
3. YOLO nano: RGB / T / 4-ch (хірургія першого conv, zero або mean init); mAP50-95 окремо для ночі (LLVIP, FLIR ніч) і дня
   (FLIR день; LLVIP має лише одну денну тестову сцену) з bootstrap-CI.
4. Абляції: T = 0, зсув T ±2 / ±4 px; приріст 4-ch над кращою одномодальною — парним bootstrap.

Критерії: спліт за сценами / поїздками (як у авторів); залишок p50 ≤ 2 px на далеких; 4-ch вхід перевірено; приріст night —
з CI або чесне «немає». Здача — архів (код + звіт) через LMS до початку заняття 13. Оцінка: коректність 40 · метрики 20 · код 20 · звіт 20.

## Залежності

`numpy>=2 scipy opencv-python matplotlib ipywidgets torch requests`; для розділу 5 Demo 3 — `ultralytics` (опційно,
перевірено на 8.4.198 і 8.4.201).

## Джерела

- Hwang et al. — Multispectral Pedestrian Detection: Benchmark Dataset and Baseline (KAIST), CVPR 2015 · sanitized-розмітка: Li et al., BMVC 2018 · KAIST-Paired: Zhang et al. (AR-CNN), ICCV 2019
- Jia et al. — LLVIP: A Visible-infrared Paired Dataset for Low-light Vision, ICCVW 2021 · [github.com/bupt-ai-cz/LLVIP](https://github.com/bupt-ai-cz/LLVIP)
- Zhang et al. — Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks (FLIR-aligned), ICIP 2020, [arXiv:2009.12664](https://arxiv.org/abs/2009.12664)
- Teledyne FLIR — ADAS Thermal Dataset v2 · [adas-dataset-v2.flirconservator.com](https://adas-dataset-v2.flirconservator.com/) · реєстрація й умови: [oem.flir.com/…/adas-dataset-form](https://www.oem.flir.com/solutions/automotive/adas-dataset-form/) · завантаження: `python tools/fetch_flir_adas.py --help`
- Liu et al. — TarDAL / M3FD, CVPR 2022 · Shin et al. — MS2, CVPR 2023 · Sun et al. — DroneVehicle, IEEE TCSVT 2022 · Anti-UAV410, T-PAMI 2023 · VTUAV, CVPR 2022
- Fang, Han, Wang — Cross-Modality Fusion Transformer for Multispectral Object Detection, [arXiv:2111.00273](https://arxiv.org/abs/2111.00273)
- Wan et al. — YOLOv11-RGBT: Towards a Comprehensive Single-Stage Multispectral Object Detection Framework, [arXiv:2506.14696](https://arxiv.org/abs/2506.14696)
- Solovyev, Wang, Gabruseva — Weighted Boxes Fusion, Image and Vision Computing 2021, [arXiv:1910.13302](https://arxiv.org/abs/1910.13302)
- Maes et al. — Multimodality Image Registration by Maximization of Mutual Information, IEEE TMI 1997
- Ultralytics 8.4 — `utils/patches.py` (imread), `nn/tasks.py` (load, перший conv), `data/augment.py`
