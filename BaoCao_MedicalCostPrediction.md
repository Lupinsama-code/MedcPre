# Báo cáo Phân tích và Dự báo Chi phí Y tế

## 1. Giới thiệu

Bài toán dự báo chi phí y tế sử dụng bộ dữ liệu insurance.csv, với mục tiêu xây dựng mô hình hồi quy dự báo chi phí dựa trên các đặc trưng như tuổi, giới tính, BMI, tình trạng hút thuốc, vùng miền, v.v. Dự án sử dụng Python, scikit-learn, pandas, matplotlib, seaborn.

## 2. Phân tích dữ liệu (EDA)

### 2.1. Phân phối biến mục tiêu (charges)
- **Histogram**: Cho thấy phân phối lệch phải, nhiều giá trị nhỏ, một số giá trị lớn (outlier).
- **Boxplot**: Xác định các giá trị ngoại lai, mức độ phân tán lớn.

### 2.2. So sánh charges theo giới tính và hút thuốc
- **Boxplot theo giới tính**: Chi phí y tế giữa nam và nữ không khác biệt nhiều.
- **Boxplot theo hút thuốc**: Người hút thuốc có chi phí y tế cao hơn rõ rệt so với người không hút thuốc.

### 2.3. Mối quan hệ tuổi và chi phí
- **Scatter plot Age vs Charges**: Tuổi càng cao, chi phí y tế có xu hướng tăng, đặc biệt với người hút thuốc.

## 3. Tiền xử lý dữ liệu
- Mã hóa biến phân loại (LabelEncoder, OneHotEncoder).
- Chia tập train/test.
- Chuẩn hóa dữ liệu nếu cần.


## 4. Dự kiến về các mô hình sử dụng

- **Linear Regression**: Mô hình tuyến tính cơ bản, dễ triển khai và giải thích. Dự kiến phù hợp nếu mối quan hệ giữa các đặc trưng và chi phí y tế gần tuyến tính, nhưng có thể bị ảnh hưởng bởi outlier và phân phối lệch.
- **Random Forest**: Mô hình ensemble dựa trên nhiều cây quyết định, có khả năng bắt các quan hệ phi tuyến và giảm overfitting so với cây đơn lẻ. Dự kiến cho kết quả tốt hơn Linear Regression, đặc biệt khi dữ liệu có nhiều đặc trưng tương tác phức tạp.
- **XGBoost**: Mô hình boosting mạnh mẽ, tối ưu hóa tốt cho các bài toán hồi quy với dữ liệu phức tạp. Dự kiến sẽ cho kết quả tốt nhất nếu dữ liệu đủ lớn và có nhiều đặc trưng quan trọng.

Việc thử nghiệm nhiều mô hình giúp so sánh hiệu quả và lựa chọn phương án tối ưu cho bài toán dự báo chi phí y tế.

## 4. Xây dựng và đánh giá mô hình
- Thử nghiệm các mô hình: Linear Regression, Random Forest, XGBoost, v.v.
- Đánh giá bằng các chỉ số: R², RMSE, MAE trên cả train và test.

### 4.1. So sánh mô hình
- **Biểu đồ cột**: So sánh trực quan R², RMSE, MAE giữa các mô hình.
- Nhận xét: Mô hình ensemble (Random Forest, XGBoost) thường cho kết quả tốt hơn Linear Regression.

## 5. Kết quả dự báo
- File predictions.csv lưu giá trị thực tế và dự báo.
- Độ lệch giữa y_true và y_pred phản ánh hiệu quả mô hình.

## 6. Kết luận
- Dữ liệu có nhiều outlier, phân phối lệch phải.
- Hút thuốc là yếu tố ảnh hưởng mạnh nhất đến chi phí y tế.
- Mô hình ensemble cho kết quả dự báo tốt nhất.

## 7. Đề xuất
- Có thể thử thêm các kỹ thuật xử lý outlier, feature engineering.
- Thử nghiệm thêm các mô hình khác hoặc tuning hyperparameter.

---

*Báo cáo tự động sinh bởi hệ thống AI. Nếu cần bản tiếng Anh hoặc bổ sung chi tiết, vui lòng liên hệ.*
