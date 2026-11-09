import shutil
from pathlib import Path

PROJECT_ROOT = Path("D:/Yolo/Vision-AI")
BASE_SUBSET = PROJECT_ROOT / "data" / "bdd100k_subset"
P_VALUES = [0, 10, 20, 30, 50]

for p in P_VALUES:
    aug_dir = PROJECT_ROOT / "data" / f"augmented_p{p}"

    # Копируем val
    val_img_src = BASE_SUBSET / "images" / "val"
    val_lbl_src = BASE_SUBSET / "labels" / "val"
    val_img_dst = aug_dir / "images" / "val"
    val_lbl_dst = aug_dir / "labels" / "val"

    if not val_img_dst.exists():
        print(f"📋 Копирование val в augmented_p{p}...")
        shutil.copytree(val_img_src, val_img_dst)
        shutil.copytree(val_lbl_src, val_lbl_dst)
    else:
        print(f"✅ augmented_p{p}/val уже существует")

    # Копируем test
    test_img_src = BASE_SUBSET / "images" / "test"
    test_lbl_src = BASE_SUBSET / "labels" / "test"
    test_img_dst = aug_dir / "images" / "test"
    test_lbl_dst = aug_dir / "labels" / "test"

    if not test_img_dst.exists():
        print(f"📋 Копирование test в augmented_p{p}...")
        shutil.copytree(test_img_src, test_img_dst)
        shutil.copytree(test_lbl_src, test_lbl_dst)
    else:
        print(f"✅ augmented_p{p}/test уже существует")

print("\n✅ Все папки val/test скопированы!")