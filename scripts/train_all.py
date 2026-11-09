from ultralytics import YOLO
from pathlib import Path
import time
import yaml

PROJECT_ROOT = Path("D:/Yolo/Vision-AI")
P_VALUES = [0, 10, 20, 30, 50]
SEEDS = [42, 123, 456]

# Параметры обучения под RTX 3060 12GB
TRAIN_ARGS = {
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": 0,  # GPU 0 (RTX 3060)
    "workers": 4,  # Потоки загрузки данных
    "verbose": False,  # Тихий режим для чистоты логов
    "patience": 20,  # Ранняя остановка если нет улучшений 20 эпох подряд
}


def create_data_yaml(dataset_dir):
    """Создает data.yaml для конкретной папки аугментации."""
    yaml_path = dataset_dir / "data.yaml"

    # Если уже существует - не перезаписываем
    if yaml_path.exists():
        return str(yaml_path)

    config = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',  # Используем общий val из bdd100k_subset
        'test': 'images/test',  # Используем общий test из bdd100k_subset
        'nc': 3,
        'names': ['pedestrian', 'rider', 'car']
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return str(yaml_path)


if __name__ == "__main__":
    print("=" * 60)
    print("МАССОВОЕ ОБУЧЕНИЕ YOLOv8s (5 долей × 3 сида = 15 моделей)")
    print("=" * 60)

    total_start = time.time()
    model_count = 0

    for p in P_VALUES:
        dataset_dir = PROJECT_ROOT / "data" / f"augmented_p{p}"

        # Создаем конфиг для этой конфигурации
        data_yaml_path = create_data_yaml(dataset_dir)
        print(f"\n📄 Создан конфиг: {data_yaml_path}")

        for seed in SEEDS:
            model_count += 1
            run_name = f"p{p}_seed{seed}"
            print(f"\n[{model_count}/15] Запуск: {run_name}")

            start_time = time.time()

            # Загружаем модель с COCO весами
            model = YOLO("yolov8s.pt")

            # Обучаем
            results = model.train(
                data=data_yaml_path,
                project="runs/detect",
                name=run_name,
                seed=seed,
                exist_ok=True,  # Не падать если папка уже есть
                **TRAIN_ARGS
            )

            elapsed = time.time() - start_time
            mAP50 = results.results_dict.get('metrics/mAP50(B)', 0)
            print(f"   ✅ Готово за {elapsed / 60:.1f} мин. | mAP50: {mAP50:.4f}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"🎉 ВСЕ 15 МОДЕЛЕЙ ОБУЧЕНЫ!")
    print(f"⏱️ Общее время: {total_elapsed / 3600:.2f} часов")
    print(f"📁 Веса сохранены в: runs/detect/")
    print("=" * 60)