# Классификация
 пошаговая инструкция для запуска, сначала устанавливает зависимости:

```pip install -r requirements.txt```
## 1. Подготовка данных
python -m src.two_towers.prepare_data
## 2. Обучение модели
python -m src.two_towers.train
## 3. Предсказание
python -m src.two_towers.predict

# Регрессия
## 1. Подготовка данных
python -m src.two_towers_regres.prepare_data
## 2. Обучение модели
python -m src.two_towers_regres.train
## 3. Предсказание
python -m src.two_towers_regres.predict


