# local_paths.py
from pathlib import Path

# 1. Программно определяем PROJECT_ROOT.
# Мы ищем директорию, в которой находится папка 'src'.
# __file__ -> путь к текущему файлу (src/paths.py)
# Path(__file__) -> объект Path
# .parent -> родительская папка (src)
# .parent -> еще раз родительская папка (корень проекта)
PROJECT_ROOT = Path(__file__).parent.parent

# 2. Теперь мы можем безопасно строить пути к другим директориям от PROJECT_ROOT.
# Эти переменные будут доступны для импорта по всему проекту.
CONFIGS_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EDIT_DATA_DIR = DATA_DIR / "edit" # Пример, если у вас есть такая папка
FINAL_DATA_DIR = DATA_DIR / "final"

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# 3. (Опционально) Вы можете добавить пути к конкретным файлам
# SOME_IMPORTANT_FILE = RAW_DATA_DIR / "some_file.csv"

# Эта проверка гарантирует, что при прямом запуске этого файла
# вы увидите пути, которые он сгенерировал. Полезно для отладки.
if __name__ == "__main__":
    print(f"Корневая директория проекта: {PROJECT_ROOT}")
    print(f"Путь к 'raw' данным: {RAW_DATA_DIR}")
    print(f"Существует ли папка 'raw_data'? {RAW_DATA_DIR.exists()}")