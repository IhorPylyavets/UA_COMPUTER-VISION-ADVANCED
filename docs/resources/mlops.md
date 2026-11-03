# MLOps, Active Learning, XAI і сервінг: інструменти на GitHub

Заняття 13–14. Тут бібліотеки й платформи, які покривають кілька методів або весь етап конвеєра.
Стан — листопад 2026: перед використанням перевір дату останнього коміту.

## Дані: версіонування, розмітка, аналіз

| Репозиторій | Що робить | Примітка |
|---|---|---|
| [treeverse/dvc](https://github.com/treeverse/dvc) | версії датасетів і моделей поруч із git, конвеєри, експерименти | Колишній `iterative/dvc`; Apache-2.0 |
| [cvat-ai/cvat](https://github.com/cvat-ai/cvat) | розмітка зображень і відео: bbox, маски, треки, автоанотація моделями | Формат GT у ДЗ 4 (MDMT); MIT |
| [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio) | розмітка будь-яких модальностей, ML-бекенд для передрозмітки | Apache-2.0 |
| [voxel51/fiftyone](https://github.com/voxel51/fiftyone) | перегляд датасету й передбачень, пошук помилок розмітки, ембеддинги, оцінка моделей | Найзручніший інструмент для data-centric аналізу |
| [cleanlab/cleanlab](https://github.com/cleanlab/cleanlab) | пошук хибних міток, дублікатів і викидів | Apache-2.0 |

## Active Learning

| Репозиторій | Методи | Примітка |
|---|---|---|
| [baal-org/baal](https://github.com/baal-org/baal) | MC Dropout, BALD, entropy, margin та ін. | Байєсівський AL для PyTorch |
| [modAL-python/modAL](https://github.com/modAL-python/modAL) | uncertainty, margin, entropy sampling, query-by-committee | Простий API у стилі scikit-learn; останній коміт 2024 |

## Пояснюваність (XAI)

| Репозиторій | Методи | Примітка |
|---|---|---|
| [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | Grad-CAM, Grad-CAM++, HiResCAM, XGradCAM, Score-CAM, Ablation-CAM, Eigen-CAM, LayerCAM, FullGrad, KPCA-CAM, FinerCAM та ін. | Працює з CNN, ViT, детекторами й сегментацією; MIT |
| [meta-pytorch/captum](https://github.com/meta-pytorch/captum) | Integrated Gradients, DeepLIFT, GradientSHAP, Occlusion, LRP та ін. | Атрибуції для будь-якої моделі PyTorch; колишній `pytorch/captum` |
| [understandable-machine-intelligence-lab/Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) | 35+ метрик якості пояснень у 6 категоріях: faithfulness, robustness, localisation та ін. | Щоб порівнювати пояснення числами, а не на око |

## Сервінг

| Репозиторій | Що робить | Примітка |
|---|---|---|
| [triton-inference-server/server](https://github.com/triton-inference-server/server) | TensorRT, ONNX, PyTorch, OpenVINO в одному сервері; динамічний батчинг, ансамблі | Стандарт для GPU-сервінгу |
| [bentoml/BentoML](https://github.com/bentoml/BentoML) | пакування моделі в сервіс, адаптивний батчинг, Docker | Apache-2.0 |
| [Lightning-AI/LitServe](https://github.com/Lightning-AI/LitServe) | легкий сервер поверх FastAPI з батчингом і стрімінгом | Найменше коду до першого API |

## Датасети для експериментів з якістю даних

| Датасет | Що в ньому | Для чого |
|---|---|---|
| [cleanlab/label-errors](https://github.com/cleanlab/label-errors) | знайдені помилки розмітки в тестах ImageNet, CIFAR, QuickDraw та ін. | Наочно: скільки «шуму» в еталонних датасетах |
| [CIFAR-10N / 100N](https://github.com/UCSC-REAL/cifar-10-100n) | CIFAR з реальними помилками людської розмітки | Бенчмарк навчання на шумних мітках і пошуку помилок |
