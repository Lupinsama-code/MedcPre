import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_target_distribution(df):
    """
    Vẽ histogram và boxplot cho biến mục tiêu 'charges'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df['charges'], bins=50, alpha=0.7, color='skyblue')
    axes[0].set_title('Phân phối Chi phí Y tế')
    axes[0].set_xlabel('Chi phí ($)')
    axes[0].set_ylabel('Tần suất')
    axes[1].boxplot(df['charges'], vert=False)
    axes[1].set_title('Boxplot Chi phí Y tế')
    plt.tight_layout()
    plt.show()

def plot_feature_boxplots(df):
    """
    Vẽ boxplot charges theo giới tính và tình trạng hút thuốc.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df.boxplot(column='charges', by='sex', ax=axes[0])
    axes[0].set_title('Chi phí theo Giới tính')
    df.boxplot(column='charges', by='smoker', ax=axes[1])
    axes[1].set_title('Chi phí theo Tình trạng Hút thuốc')
    plt.suptitle('')
    plt.tight_layout()
    plt.show()

def plot_scatter_age_charges(df):
    """
    Vẽ scatter plot Age vs Charges.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(df['age'], df['charges'], alpha=0.6, color='coral')
    plt.title('Tuổi vs Chi phí')
    plt.xlabel('Tuổi')
    plt.ylabel('Chi phí ($)')
    plt.show()

def plot_model_comparison(comparison_df):
    """
    Vẽ so sánh các chỉ số đánh giá mô hình.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['Train R²', 'Test R²']
    x = np.arange(len(comparison_df.index))
    width = 0.35
    for i, metric in enumerate(metrics):
        axes[0, 0].bar(x + i*width - width/2, comparison_df[metric], width, label=metric, alpha=0.8)
    axes[0, 0].set_title('So sánh R² Score')
    axes[0, 0].set_xlabel('Mô hình')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(comparison_df.index, rotation=45)
    axes[0, 0].legend()
    rmse_metrics = ['Train RMSE', 'Test RMSE']
    for i, metric in enumerate(rmse_metrics):
        axes[0, 1].bar(x + i*width - width/2, comparison_df[metric], width, label=metric, alpha=0.8)
    axes[0, 1].set_title('So sánh RMSE')
    axes[0, 1].set_xlabel('Mô hình')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(comparison_df.index, rotation=45)
    axes[0, 1].legend()
    axes[1, 0].bar(comparison_df.index, comparison_df['Test R²'], color='green', alpha=0.7)
    axes[1, 0].set_title('Test R² Score - So sánh')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 1].bar(comparison_df.index, comparison_df['Test MAE'], color='orange', alpha=0.7)
    axes[1, 1].set_title('Test MAE - So sánh')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
