import pandas as pd
from src.data_utils import load_or_create_data, preprocess_data
from src.model_utils import get_models, train_and_evaluate, results_to_dataframe
from src.visualization import plot_target_distribution, plot_feature_boxplots, plot_scatter_age_charges, plot_model_comparison
import joblib
import os

def main():
    import sys
    save_mode = '--save' in sys.argv

    # 1. Load hoặc tạo dữ liệu
    df = load_or_create_data('data/insurance.csv', 'data/insurance_sample.csv')
    print('--- Dữ liệu mẫu ---')
    print(df.head())

    # 2. EDA
    plot_target_distribution(df)
    plot_feature_boxplots(df)
    plot_scatter_age_charges(df)

    # 3. Tiền xử lý
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Train & evaluate
    models = get_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    comparison_df = results_to_dataframe(results)
    print(comparison_df.round(4))
    plot_model_comparison(comparison_df)

    # 5. Lưu kết quả nếu có --save
    if save_mode:
        best_model_name = comparison_df['Test R²'].idxmax()
        best_model = results[best_model_name]['Model']
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        X_test.to_csv('data/X_test.csv', index=False)
        y_test.to_csv('data/y_test.csv', index=False)
        y_pred = best_model.predict(X_test)
        pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_csv('data/predictions.csv', index=False)
        joblib.dump(best_model, 'models/best_model.pkl')
        print(f'✅ Đã lưu X_test, y_test, predictions.csv, best_model.pkl. Best model: {best_model_name}')
    else:
        print("(Không lưu file. Thêm --save để lưu dữ liệu và model)")

if __name__ == "__main__":
    main()
