import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def main():
    print("Loading data...")
    start_load = time.time()
    # Đường dẫn file csv trên VM Linux
    try:
        df = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("Error: creditcard.csv not found. Please run the kaggle download command first.")
        return

    load_time = time.time() - start_load
    print(f"Data loaded in {load_time:.2f} seconds.")

    # Chuẩn bị dữ liệu
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training LightGBM model on CPU (n2-standard-8)...")
    start_train = time.time()
    # verbose=-1 để giảm bớt log không cần thiết
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"Model trained in {train_time:.2f} seconds.")

    # Dự đoán và tính toán Metric
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"AUC-ROC: {auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # Đo độ trễ Inference (1 dòng và 1000 dòng)
    single_row = X_test.iloc[[0]]
    start_inf_1 = time.time()
    model.predict(single_row)
    latency_1 = time.time() - start_inf_1

    batch_rows = X_test.head(1000)
    start_inf_1000 = time.time()
    model.predict(batch_rows)
    throughput_1000 = time.time() - start_inf_1000

    results = {
        "load_data_time_sec": round(load_time, 4),
        "training_time_sec": round(train_time, 4),
        "auc_roc": round(auc, 4),
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "inference_latency_1_row_sec": round(latency_1, 6),
        "inference_latency_1000_rows_sec": round(throughput_1000, 6)
    }

    with open('benchmark_result.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Metrics saved to benchmark_result.json")

if __name__ == "__main__":
    main()
