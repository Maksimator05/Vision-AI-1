import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import cv2

# Фиксируем сиды для воспроизводимости
random.seed(42)
np.random.seed(42)

PROJECT_ROOT = Path("D:/Yolo/Vision-AI")
BASE_TRAIN_IMG = PROJECT_ROOT / "data" / "bdd100k_subset" / "images" / "train"
BASE_TRAIN_LBL = PROJECT_ROOT / "data" / "bdd100k_subset" / "labels" / "train"

# Значения доли аугментаций из протокола
P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]

# Уровень интенсивности для обучения (фиксированный средний уровень 3)
TRAIN_LEVEL = 3

# Параметры Albumentations для уровня 3 (согласно протоколу)
FOG_L3 = {"fog_lower": 0.4, "fog_upper": 0.5}
RAIN_L3 = {"drop_length": 16, "drop_width": 2, "blur_value": 5, "brightness_coeff": 0.7}
SNOW_L3 = {"snow_point_lower": 0.3, "snow_point_upper": 0.4, "brightness_coeff": 2.1}


def get_weather_transform(weather_type):
    """Возвращает трансформацию для конкретного типа погоды."""
    if weather_type == "fog":
        return A.RandomFog(p=1.0, **FOG_L3)
    elif weather_type == "rain":
        return A.RandomRain(p=1.0, **RAIN_L3)
    elif weather_type == "snow":
        return A.RandomSnow(p=1.0, **SNOW_L3)


def apply_augmentation(img_path, weather_type, out_path):
    """Применяет аугментацию к изображению и сохраняет результат."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Не удалось прочитать: {img_path}")
        return False

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_weather_transform(weather_type)
    augmented = transform(image=img)["image"]
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), augmented)
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("СОЗДАНИЕ АУГМЕНТИРОВАННЫХ ОБУЧАЮЩИХ НАБОРОВ")
    print("=" * 60)

    # Получаем список всех файлов в базовом train
    all_files = sorted(BASE_TRAIN_IMG.glob("*.jpg"))
    total_images = len(all_files)
    print(f"📁 Базовый train: {total_images} изображений")

    for p in P_VALUES:
        p_int = int(p * 100)
        out_img_dir = PROJECT_ROOT / "data" / f"augmented_p{p_int}" / "images" / "train"
        out_lbl_dir = PROJECT_ROOT / "data" / f"augmented_p{p_int}" / "labels" / "train"

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        n_aug = int(total_images * p)  # Количество изображений для аугментации

        # Детерминированный выбор изображений
        shuffled_files = all_files.copy()
        random.shuffle(shuffled_files)
        aug_files = shuffled_files[:n_aug]
        clean_files = shuffled_files[n_aug:]

        print(f"\n🔧 Обработка p={p} ({p_int}%):")
        print(f"   Чистых: {len(clean_files)}, Аугментированных: {n_aug}")

        # 1. Копируем чистые изображения
        for f in tqdm(clean_files, desc="Копирование чистых", leave=False):
            shutil.copy(f, out_img_dir / f.name)
            shutil.copy(BASE_TRAIN_LBL / (f.stem + ".txt"), out_lbl_dir / (f.stem + ".txt"))

        # 2. Применяем аугментации (равномерное распределение: fog/rain/snow)
        weather_types = ["fog", "rain", "snow"]
        success_count = 0
        fail_count = 0

        for i, f in enumerate(tqdm(aug_files, desc=f"Аугментация (p={p})", leave=False)):
            weather = weather_types[i % 3]  # Равномерное распределение
            out_path = out_img_dir / f.name

            if apply_augmentation(f, weather, out_path):
                shutil.copy(BASE_TRAIN_LBL / (f.stem + ".txt"), out_lbl_dir / (f.stem + ".txt"))
                success_count += 1
            else:
                fail_count += 1

        print(f"   ✅ Успешно: {success_count}, ❌ Ошибок: {fail_count}")

    print("\n" + "=" * 60)
    print("✅ ВСЕ НАБОРЫ СОЗДАНЫ!")
    print("=" * 60)
    print("Структура:")
    for p in P_VALUES:
        p_int = int(p * 100)
        path = PROJECT_ROOT / "data" / f"augmented_p{p_int}"
        print(f"   📁 {path}")
    print("=" * 60)