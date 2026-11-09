import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Фиксируем сид для воспроизводимости разбиения
random.seed(42)

# === АБСОЛЮТНЫЕ ПУТИ ПОД ТВОЮ СТРУКТУРУ ===
PROJECT_ROOT = Path("D:/Yolo/Vision-AI")

# Пути к BDD100K
BDD_LABELS_DIR = PROJECT_ROOT / "data" / "bdd100k_labels_release" / "bdd100k" / "labels"
BDD_IMAGES_ROOT = PROJECT_ROOT / "data" / "bdd100k" / "bdd100k" / "images" / "10k"

# Выходная папка
OUT_DIR = PROJECT_ROOT / "data" / "bdd100k_subset"

# Целевые классы
TARGET_CLASSES = {"pedestrian", "rider", "car"}
CLASS_TO_IDX = {c: i for i, c in enumerate(sorted(TARGET_CLASSES))}

# Стандартный размер BDD100K
IMG_W, IMG_H = 1280, 720


def bdd_to_yolo(box):
    """Конвертация [x1, y1, x2, y2] в нормированный YOLO формат."""
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    w = x2 - x1
    h = y2 - y1
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    return f"{xc / IMG_W:.6f} {yc / IMG_H:.6f} {w / IMG_W:.6f} {h / IMG_H:.6f}"


def build_image_map(root_dir):
    """Рекурсивно сканирует папку и создает словарь {имя_файла.jpg: полный_путь}"""
    image_map = {}
    print(f"🔍 Сканирование папки изображений: {root_dir}")
    for img_path in root_dir.rglob("*.jpg"):
        image_map[img_path.name] = img_path
    print(f"✅ Найдено всего изображений: {len(image_map)}")
    return image_map


def process_split(annotations, split_name, out_images_dir, out_labels_dir, image_map):
    """Обработка одного сплита с использованием карты изображений."""
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    kept_count = 0
    missing_count = 0

    for item in tqdm(annotations, desc=f"Обработка {split_name}"):
        # Фильтрация по целевым классам
        valid_labels = [l for l in item.get("labels", []) if l.get("category") in TARGET_CLASSES]
        if not valid_labels:
            continue

        img_name = item["name"]

        # Ищем картинку по имени в нашей карте
        img_path = image_map.get(img_name)

        if not img_path:
            missing_count += 1
            continue

        # Копируем изображение
        shutil.copy(img_path, out_images_dir / img_name)

        # Записываем метки в YOLO-формате
        label_path = out_labels_dir / (Path(img_name).stem + ".txt")
        with open(label_path, "w") as f:
            for label in valid_labels:
                cls_idx = CLASS_TO_IDX[label["category"]]
                yolo_box = bdd_to_yolo(label["box2d"])
                f.write(f"{cls_idx} {yolo_box}\n")

        kept_count += 1

    print(f"[{split_name}] Сохранено: {kept_count}, Пропущено (нет картинки): {missing_count}")
    return kept_count


if __name__ == "__main__":
    print("=" * 60)
    print("ПОДГОТОВКА ДАТАСЕТА BDD100K ДЛЯ YOLOv8")
    print("=" * 60)

    # Проверка путей
    train_json = BDD_LABELS_DIR / "bdd100k_labels_images_train.json"
    val_json = BDD_LABELS_DIR / "bdd100k_labels_images_val.json"

    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError(f"❌ JSON файлы не найдены в: {BDD_LABELS_DIR}")
    if not BDD_IMAGES_ROOT.exists():
        raise FileNotFoundError(f"❌ Папка изображений не найдена: {BDD_IMAGES_ROOT}")

    print("\nЗагрузка аннотаций...")
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_json, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    print(f"Train аннотаций: {len(train_data)}, Val аннотаций: {len(val_data)}")

    # Строим карту изображений ОДИН РАЗ для всех сплитов
    image_map = build_image_map(BDD_IMAGES_ROOT)

    # Разбиваем val на val и test (50/50)
    random.shuffle(val_data)
    mid = len(val_data) // 2
    val_split = val_data[:mid]
    test_split = val_data[mid:]

    print(f"\nРазбиение: Train={len(train_data)}, Val={len(val_split)}, Test={len(test_split)}")

    # Обработка всех сплитов
    n_train = process_split(train_data, "train", OUT_DIR / "images" / "train", OUT_DIR / "labels" / "train", image_map)
    n_val = process_split(val_split, "val", OUT_DIR / "images" / "val", OUT_DIR / "labels" / "val", image_map)
    n_test = process_split(test_split, "test", OUT_DIR / "images" / "test", OUT_DIR / "labels" / "test", image_map)

    # Создаем data.yaml
    yaml_content = f"""path: {OUT_DIR.absolute()}
train: images/train
val: images/val
test: images/test

nc: 3
names: ['pedestrian', 'rider', 'car']
"""
    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n{'=' * 60}")
    print(f"✅ ДАТАСЕТ УСПЕШНО ПОДГОТОВЛЕН!")
    print(f"{'=' * 60}")
    print(f"📁 Папка датасета: {OUT_DIR}")
    print(f"📊 Статистика:")
    print(f"   Train: {n_train} изображений")
    print(f"   Val:   {n_val} изображений")
    print(f"   Test:  {n_test} изображений")
    print(f"{'=' * 60}")