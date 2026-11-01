# Foundation models: збірки моделей на GitHub

Заняття 5–6, треки 1–2 фінального проєкту. Тут лише репозиторії, де під одним API зібрано багато моделей.
Стан — листопад 2026: перед використанням перевір дату останнього коміту й ліцензію ваг: вона часто суворіша за ліцензію коду.

## Бекбони та ембеддинги

| Репозиторій | Моделі | Примітка |
|---|---|---|
| [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) (timm) | сотні бекбонів: ViT, DeiT, ConvNeXt, EVA, DINOv2, SigLIP та ін. | Найпростіший спосіб узяти бекбон під свою задачу; Apache-2.0 |
| [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | CLIP, SigLIP, EVA-CLIP, MetaCLIP, десятки чекпойнтів | Відкриті відтворення CLIP з навчанням |
| [huggingface/transformers](https://github.com/huggingface/transformers) | CLIP, SigLIP 2, DINOv2/v3, OWLv2, Grounding DINO, Florence-2, Qwen-VL та ін. | Єдиний API для більшості моделей заняття 5–6 |
| [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | DINOv3 ViT і ConvNeXt, голови для сегментації й глибини | Gated-ліцензія Meta: прийняти на Hugging Face |

## Open-vocabulary детекція та сегментація

| Репозиторій | Моделі | Примітка |
|---|---|---|
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | YOLO, YOLO-World, YOLOE, SAM / SAM 2, FastSAM, MobileSAM, RT-DETR | Найшвидший старт; AGPL-3.0 — для закритого продукту потрібна комерційна ліцензія |
| [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Grounding DINO, GLIP, DINO, DETR-сімейство, Mask R-CNN та сотні конфігів | Не розвивається з 2024, але найповніший зоопарк детекторів; Apache-2.0 |
| [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | Grounding DINO / Florence-2 + SAM 2 для зображень і відео | Готовий конвеєр «текст → маски → трек» |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2), [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | SAM 2.1 (promptable, відео з пам'яттю); SAM 3 (концепт → усі екземпляри з ID) | SAM 3 — gated-ліцензія Meta |

## Інструменти навколо моделей

[roboflow/supervision](https://github.com/roboflow/supervision) — спільні типи детекцій, анотатори, трекери, метрики й SAHI-слайсинг для будь-якої моделі. MIT.
