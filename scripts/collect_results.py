import pandas as pd
from pathlib import Path
import yaml

PROJECT_ROOT = Path("D:/Yolo/Vision-AI")
RUNS_DIR = PROJECT_ROOT / "runs" / "detect" / "runs" / "detect"
P_VALUES = [0, 10, 20, 30, 50]
SEEDS = [42, 123, 456]

results = []

for p in P_VALUES:
    for seed in SEEDS:
        run_name = f"p{p}_seed{seed}"
        results_csv = RUNS_DIR / run_name / "results.csv"

        if not results_csv.exists():
            print(f"⚠️ Не найден: {results_csv}")
            continue

        # Читаем последнюю строку (best epoch)
        df = pd.read_csv(results_csv)
        last_row = df.iloc[-1]

        # Извлекаем метрики (названия колонок могут отличаться в разных версиях)
        mAP50 = last_row.get('metrics/mAP50(B)', last_row.get('metrics/mAP50', 0))
        mAP50_95 = last_row.get('metrics/mAP50-95(B)', last_row.get('metrics/mAP50-95', 0))
        precision = last_row.get('metrics/precision(B)', last_row.get('metrics/precision', 0))
        recall = last_row.get('metrics/recall(B)', last_row.get('metrics/recall', 0))

        results.append({
            'p': p,
            'seed': seed,
            'mAP50': round(mAP50, 4),
            'mAP50-95': round(mAP50_95, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4)
        })

# Создаем DataFrame и сохраняем
df_results = pd.DataFrame(results)
df_results.to_csv(PROJECT_ROOT / "results/all_metrics.csv", index=False)

# Выводим сводную таблицу (среднее по сидам)
summary = df_results.groupby('p')[['mAP50', 'mAP50-95', 'Precision', 'Recall']].agg(['mean', 'std']).round(4)
print("\n" + "=" * 70)
print("📊 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (Val Set)")
print("=" * 70)
print(summary)
print("=" * 70)
print(f"\n💾 Полные результаты сохранены в: results/all_metrics.csv")