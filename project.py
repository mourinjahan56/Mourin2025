import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

def bilingual_print(msg_en, msg_bn, lang='en'):
    print(msg_en if lang == 'en' else msg_bn)

def detect_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data[column] < lower) | (data[column] > upper)]

def analyze_chunk(chunk, target, lang='en', model_type='sklearn', output_dir='.'):
    # Select only numeric columns
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
    if target not in numeric_cols:
        bilingual_print(
            f"Error: Target column '{target}' is not numeric or missing.",
            f"ত্রুটি: টার্গেট কলাম '{target}' সংখ্যাসূচক নয় বা অনুপস্থিত।",
            lang
        )
        return None

    features = [col for col in numeric_cols if col != target]
    if not features:
        bilingual_print(
            "No numeric features found for regression.",
            "রিগ্রেশন করার জন্য কোনো সংখ্যাসূচক ফিচার পাওয়া যায়নি।",
            lang
        )
        return None

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = chunk[features]
    y = chunk[target]
    X_imputed = imputer.fit_transform(X)
    y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Fit regression model
    if model_type == 'tensorflow':
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(X_imputed.shape[1],))
        ])
        tf_model.compile(optimizer='adam', loss='mse')
        tf_model.fit(X_imputed, y_imputed, epochs=10, verbose=0)
        y_pred = tf_model.predict(X_imputed).ravel()
    else:
        model = LinearRegression()
        model.fit(X_imputed, y_imputed)
        y_pred = model.predict(X_imputed)

    # Error metrics
    mse = mean_squared_error(y_imputed, y_pred)
    mae = mean_absolute_error(y_imputed, y_pred)
    r2 = r2_score(y_imputed, y_pred)

    bilingual_print(
        f"Metrics: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}",
        f"মেট্রিক্স: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}",
        lang
    )

    # Outlier detection
    outliers = detect_outliers(chunk, target)
    bilingual_print(
        f"Detected {len(outliers)} outliers. Indices: {outliers.index.tolist()}",
        f"{len(outliers)}টি আউটলায়ার সনাক্ত হয়েছে। ইনডেক্স: {outliers.index.tolist()}",
        lang
    )
    outlier_path = os.path.join(output_dir, 'outliers.csv')
    outliers.to_csv(outlier_path, index=True)

    # Visualization
    plt.figure(figsize=(6, 4))
    plt.scatter(y_imputed, y_pred, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_plot.png'))
    plt.close()

    # Residual plot
    plt.figure(figsize=(6, 4))
    plt.hist(y_imputed - y_pred, bins=30, alpha=0.7)
    plt.title('Residuals Histogram')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_hist.png'))
    plt.close()

    # Advice
    if r2 < 0.5:
        bilingual_print(
            "Model performance is low. Consider feature engineering or more data.",
            "মডেলের পারফরম্যান্স কম। আরও ডেটা বা ফিচার ইঞ্জিনিয়ারিং বিবেচনা করুন।",
            lang
        )

    # Export report
    report = pd.DataFrame({
        'Actual': y_imputed,
        'Predicted': y_pred
    })
    report_path = os.path.join(output_dir, 'regression_report.csv')
    report.to_csv(report_path, index=False)

    return {'mse': mse, 'mae': mae, 'r2': r2, 'outliers': len(outliers)}

def main():
    parser = argparse.ArgumentParser(
        description='Smart Bilingual Regression Analyzer (Bengali/English) for Big Data'
    )
    parser.add_argument('--file', required=True, help='CSV file path')
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--lang', default='en', choices=['en', 'bn'], help='Language: en or bn')
    parser.add_argument('--model', default='sklearn', choices=['sklearn', 'tensorflow'], help='Regression model')
    parser.add_argument('--output', default='.', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        bilingual_print(
            f"File '{args.file}' not found.",
            f"ফাইল '{args.file}' পাওয়া যায়নি।",
            args.lang
        )
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    metrics_list = []

    # Estimate number of rows for progress bar
    try:
        total_rows = sum(1 for _ in open(args.file)) - 1
    except Exception:
        total_rows = None

    bilingual_print("Starting analysis...", "বিশ্লেষণ শুরু হচ্ছে...", args.lang)
    for chunk in tqdm(pd.read_csv(args.file, chunksize=100000), total=(total_rows // 100000 + 1 if total_rows else None)):
        metrics = analyze_chunk(chunk, args.target, args.lang, args.model, args.output)
        if metrics:
            metrics_list.append(metrics)

    # Summary
    if metrics_list:
        avg_mse = np.mean([m['mse'] for m in metrics_list])
        avg_mae = np.mean([m['mae'] for m in metrics_list])
        avg_r2 = np.mean([m['r2'] for m in metrics_list])
        total_outliers = sum(m['outliers'] for m in metrics_list)
        bilingual_print(
            f"\nSummary:\nAvg MSE={avg_mse:.4f}, Avg MAE={avg_mae:.4f}, Avg R2={avg_r2:.4f}, Total Outliers={total_outliers}",
            f"\nসারাংশ:\nগড় MSE={avg_mse:.4f}, গড় MAE={avg_mae:.4f}, গড় R2={avg_r2:.4f}, মোট আউটলায়ার={total_outliers}",
            args.lang
        )
    else:
        bilingual_print("No valid data processed.", "কোনো বৈধ ডেটা প্রক্রিয়া করা হয়নি।", args.lang)

if __name__ == '__main__':
    main()
