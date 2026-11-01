# Генеративні моделі та синтетичні дані: збірки на GitHub

Заняття 7–8, трек 2 фінального проєкту. Тут лише репозиторії, де під одним API зібрано багато моделей або методів.
Стан — листопад 2026: перед використанням перевір дату останнього коміту й ліцензію ваг.

## Дифузія та донавчання

| Репозиторій | Що всередині | Примітка |
|---|---|---|
| [huggingface/diffusers](https://github.com/huggingface/diffusers) | SD 1.5 / SDXL / SD3, FLUX, ControlNet, IP-Adapter, inpainting, scheduler-и; скрипти LoRA і DreamBooth | Основний вибір для коду; Apache-2.0 |
| [huggingface/peft](https://github.com/huggingface/peft) | LoRA, IA³ та десятки інших адаптерів | Адаптери для diffusers і transformers |
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | LoRA, DreamBooth, textual inversion для SD 1.x / SDXL / SD3 / FLUX.1 / Lumina / HunyuanImage | Де-факто стандарт для навчання LoRA |
| [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) | LoRA для FLUX.1 / FLUX.2 та інших моделей | Простіший конфіг, ніж у kohya |
| [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) | вузловий граф: генерація, ControlNet, inpainting, апскейл, тисячі кастомних вузлів | Зручно для прототипу конвеєра синтезу; GPL-3.0 |
| [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | оригінальні ControlNet: canny, depth, seg, pose та ін. | Еталон; на практиці беруть через diffusers. Останній коміт — 2024 |

## Синтез даних рендерингом

| Репозиторій | Що всередині | Примітка |
|---|---|---|
| [DLR-RM/BlenderProc](https://github.com/DLR-RM/BlenderProc) | процедурні сцени в Blender: bbox, маски, глибина, нормалі, формат COCO / BOP | Коли потрібна точна розмітка з коробки; GPL-3.0 |
