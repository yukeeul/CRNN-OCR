# CRNN-OCR: Проект для курса ml_ops

# Что происходит
Обучаю простую модельку(CRNN) для OCR на датасете IAM-lines на torch_lightning. Изначально я планировал больше датасетов и более сложные модели, но оказалось, что это задача не уровня учебного проекта(нужно много разнообразных данных, синтетика и gpu-часы).

На вход поступает поступает одноканальное изображение, на выходе получается распознанный текст.

Используемые метрики: Accuracy и 1-NED(посимвольная точность)

Данные сохраняются в dvc на Яндекс диске, модели конвертируются в onnx, гиперпараметры настравиваются через Hydra, логи отправляются в MlFlow

# Setup
Копируем репозиторий

`git clone https://github.com/yukeeul/CRNN-OCR && cd CRNN-OCR`

Устанавливаем через poetry

`poetry install`

Переходим в созданное виртуальное окружение

`source $(poetry env info --path)/bin/activate`

Проверим pre-commit хуки

`pre-commit install`

`pre-commit run -a`

# Взаимодействие с кодом
## Train

Перед запуском нунжно поднять сервер MlFlow на 127.0.0.1:8080, в противном случае будет блокирующее ожидание

Запуск через

`python3 train.py`

Все гиперпараметры лежат в configs/main.yaml

По умолчанию датасет загружается из dvc, если это не будет работать(например, потому что я не отдам пароль), можно прописать `load_from_dvc: false`

При запуске скрипта будет с 0 обучена модель, переведена в формат onnx и сохранена в dvc(соединение с dvc работает через раз). Модель залогирует свои гиперпараметры, хэш коммита, лоссы и метрики на валидационной выборке.

В конце обученная модель .onnx будет сохранена в models/ . В терминале можно будет увидеть ее id

## Inference

`python3 inference.py +model_id=<model_id> +imgs_path=examples`

Скрипту нужно передать два параметра: id обученной локальной модели и путь до папки, содержащей тестовые изображения. В результате выполнения скрипта для каждой картинки будет вывыден предсказанный текст в формате:

```
img: examples/img1.jpg
ans: Quick brown fox

img: examples/img2.jpg
ans: jumps over lazy frog
```

## Нереализованные фичи
TensorRT, Inference Server
