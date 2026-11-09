import json
from pathlib import Path

# Пути к твоим файлам (адаптировано под твою структуру)
BDD_TRAIN_JSON = Path("data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json")
ACDC_DET_JSON = Path("data/gt_detection_trainval/instancesonly_train_gt_detection.json")


def inspect_json(file_path, name):
    print(f"\n{'=' * 50}")
    print(f"Анализ файла: {name}")
    print(f"Путь: {file_path}")
    print(f"{'=' * 50}")

    if not file_path.exists():
        print("❌ Файл не найден! Проверь путь.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Тип корня: {type(data)}")
    if isinstance(data, list):
        print(f"Количество элементов (изображений): {len(data)}")
        if len(data) > 0:
            print("\n--- Пример первого элемента (первое изображение) ---")
            print(json.dumps(data[0], indent=2, ensure_ascii=False)[:1000] + "...")  # Выводим первые 1000 символов

            # Ищем ключи
            print("\n--- Доступные ключи в элементе ---")
            print(list(data[0].keys()))

            # Если есть labels, смотрим их структуру
            if 'labels' in data[0]:
                print("\n--- Пример первой метки (label) ---")
                print(json.dumps(data[0]['labels'][0], indent=2, ensure_ascii=False))

    elif isinstance(data, dict):
        print(f"Ключи словаря: {list(data.keys())}")
        # Для COCO-подобного формата (ACDC часто такой)
        if 'images' in data:
            print(f"Количество изображений: {len(data['images'])}")
            print("\n--- Пример изображения ---")
            print(json.dumps(data['images'][0], indent=2, ensure_ascii=False))
        if 'annotations' in data:
            print(f"Количество аннотаций: {len(data['annotations'])}")
            print("\n--- Пример аннотации ---")
            print(json.dumps(data['annotations'][0], indent=2, ensure_ascii=False))
        if 'categories' in data:
            print("\n--- Категории ---")
            print(json.dumps(data['categories'], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    inspect_json(BDD_TRAIN_JSON, "BDD100K Train")
    inspect_json(ACDC_DET_JSON, "ACDC Detection Train")