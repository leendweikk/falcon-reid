> **📝 Примечание / Note:**  
> This README is translated from English by international team members. We apologize if any phrasing feels awkward or technical terms are unclear — Russian is not our first language. The English version is the authoritative source.
>
> _Этот README переведён с английского международной командой. Извините, если какие-то фразы звучат странно или технические термины неясны — русский не наш первый язык. Английская версия является авторитетным источником._

---

# Falcon ReID — переидентификация автомобилей без номерных знаков

ЛЦТ 2026 · Задача 7 «ФАЛЬКОН.Tech» — сервис, который создаёт визуальный «цифровой отпечаток» автомобиля и находит один и тот же физический автомобиль на фотографиях разных камер без использования номерного знака. Ранжирует галерею для каждого запроса и отказывает («совпадение не найдено»), когда автомобиля нет в базе.

Краткое резюме на русском. Сервис формирует признак по кропу из рамки BBox и ищет тот же автомобиль на снимках разных камер без номера. Двухэтапный поиск: быстрая модель DINOv3 ViT-B/16 (замеряемая функция extract(), ≈27 мс) отбирает 100 лучших кандидатов, ансамбль ConvNeXt-Base + ViT-B переупорядочивает их, затем k-reciprocal re-ranking. Отказ: сигнал cos+gap с порогом 0.809, обоснованным на двух независимых валидационных сплитах. Результаты (300 скрытых машин): mAP@10 82.25 / 80.29, оценка отказа 0.967. Одна команда в Docker без интернета, веса проверены по sha256.

## Результаты

Все числа получены с официальным evaluate.py на двух независимых валидационных сплитах (по 300 автомобилей, никогда не видевших во время обучения).

| Метрика | Split 42 | Split 7 |
|---|---|---|
| **mAP@10** (основная, 45%) | 82.25 | 80.29 |
| Rank-1 / Rank-5 | 78.11 / 93.99 | 75.77 / 92.26 |
| Отказ: F1 / TNR (порог 0.809) | 0.955 / 0.994 | 0.963 / 0.974 |
| Взвешенная оценка отказа (20% open-set) | 0.967 | 0.967 |
| PR-AUC уверенности | 0.996 | 0.995 |

| Скорость | Ноутбук RTX 4050 | Docker на Windows | Colab T4 |
|---|---|---|---|
| Задержка (мс) | 27.2 | 37.5 | 36.6 |
| FPS (лучший batch) | 135 | 105 | 47 |
| Очки за скорость | 20/20 | 20/20 | 10/20 |

## Быстрый старт

### 1. Официальные файлы (одна команда)
```bash
docker build --target infer -t falcon-reid .
docker run --rm --gpus all --network none \
    -v /path/to/dataset:/data:ro -v /path/to/output:/out falcon-reid
```
Датасет: images/, test_query.csv, test_gallery.csv. Выход: submission.csv, embeddings.npy, candidates.csv.

### 2. Веб-сервис
```bash
docker compose up --build
# http://localhost:8080
```

### 3. Замеренная функция
```python
from falcon import extract
vec = extract("frame.jpg", (x, y, w, h))  # (768,) L2-normalized
```

### 4. Без Docker
```bash
pip install -r requirements-infer.txt
python tools/fetch_weights.py
python -m falcon.predict --images data/images --query data/test_query.csv \
                         --gallery data/test_gallery.csv --out output
```

## Архитектура

**Этап 1 (замеряемый):** DINOv3 ViT-B/16 извлекает 768-мерный вектор из кропа. Это extract(), что замеряет таймер.

**Поиск:** косинусное сходство → топ-100 кандидатов.

**Этап 2 (не замеряется):** ансамбль (ConvNeXt-Base + ViT-B с flip TTA) переупорядочивает топ-100, затем k-reciprocal re-ranking.

**Отказ:** если уверенность топ-1 < 0.809 → нет ответа.

## Методология

**Модель.** DINOv3 ViT-B/16 (этап 1) + ConvNeXt-Base (этап 2), файнтюнены на переидентификацию.

**Обучение.** ID cross-entropy + batch-hard triplet, PK-sampling (16 машин × 4 фото), augmentation (flip, crop, mild colour jitter, random erasing). 30 epochs, AdamW.

**Валидация.** 300 скрытых машин, 50 из них — «незнакомцы» (нет в галерее). Официальный junk-фильтр (одна камера + один ID = исключить). Два сплита (seed 42 и 7) — изменение сохраняется, если выигрывает на обоих.

**Порог отказа 0.809.** Найден как середина пересечения плато на обоих сплитах (0.779–0.839). Не переобучен на одно разбиение.

## Скорость

Горлышко — CPU JPEG-декодирование, не GPU. На ноутбуке с 12 потоками CPU: 27.2 мс, 135 FPS. На Colab T4 (2 потока) — 36.6 мс, 47 FPS (ограничено CPU). Оба ≤ 40 мс, оба ≥ 20 очков.

## Безопасность номера

Тест: номера в 61% фото закрашены серым. Результат на двух сплитах: 79.14→79.64, 78.00→77.71 (в пределах шума, нет падения). Grad-CAM подтверждает: модель смотрит на кузов, свет, колёса, наклейки, не на номер.

## Анализ ошибок

**6% ошибок.** 91% — двойники (та же марка/модель/цвет). Остальное: тёмные фото, бики, ночной блик, две машины в одной рамке, ошибки разметки.

Точность по условиям: тёмные фото 0.764 mAP@10, яркие 0.860. Средние размеры сложнее (0.80 vs 0.83–0.85).

## Воспроизведение

```bash
python make_crops.py
python make_val_split.py 42 && python make_val_split.py 7
python train_final.py --model vit --lr-backbone 5e-5 --stop-epoch 20
python tools/export_weights.py --model vit --src runs/final_vit_e20/vit.pth --dst weights/vit_infer.pth
```

## Библиотеки

**Инференс:** PyTorch 2.14.0, timm 1.0.30, numpy 2.4.6, PIL, huggingface_hub.

**Сервис:** FastAPI 0.141.1, PostgreSQL + pgvector, nginx.

**Обучение:** scikit-learn, scipy, tqdm.

## Соответствие правилам

Все проверено против ТЗ:
- ✓ Никакие признаки номера не используются
- ✓ Каждый запрос обработан отдельно (нет query expansion)
- ✓ Офлайн-запуск, веса проверены sha256
- ✓ Веса ≤ 2 ГБ
- ✓ Форматы соответствуют example_submission.zip

Полные проверки в docs/RULES_CHECKLIST.md и docs/ORGANIZER_ANSWERS.md.

## Структура репозитория

- **falcon/** — инференс (extract, search, predict)
- **reid/** — обучение (модель, loss, sampler, evaluate.py)
- **service/** — веб-сервис (FastAPI, PostgreSQL, nginx)
- **weights/** — веса DINOv3 с хешами
- **docs/** — эксперименты, анализ ошибок, Grad-CAM, скорость
- **submission/** — официальные файлы для теста

## Внешние ресурсы

**Веса:** DINOv3 от Meta (претренировано на 1.7B изображений), лицензия в weights/LICENSE_DINOv3.md.

**Данные:** только датасет организаторов. VeRi-776 протестирован и отклонен (нет прироста).

**Методы:** DINOv3, Bag of Tricks, batch-hard triplet, k-reciprocal re-ranking, Grad-CAM.
