### Описание

Данный проект предназначен для изучения возможностей `SL` + `RL` и их возможностях совместной работы.
Проект является попыткой создания ИИ, более совершенного чем человечество
Как мы знаем, именно `RL` модели способны к обучению, и, по сути, способны обучаться бесконечно, даже могут стать умнее людей

### Source

- [Stocks-Daily-Price (Hugging Face)](https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price/tree/main/data)

### Venv

```
cd env
python3.11 -m pip install virtualenv
python3.11 -m venv ml_env
source ml_env/bin/activate
pip3.11 install -r reqs.req.txt
python -m ipykernel install --user --name=project-venv --display-name "ML-torch env"
```

> TensorFlow-Metal — подключаемый модуль для TensorFlow, который позволяет ускорить обучение моделей машинного обучения на графических процессорах (GPU) с помощью фреймворка Metal от Apple
