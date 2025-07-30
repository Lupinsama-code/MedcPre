import numpy as np
import pandas as pd

def load_or_create_data(filepath='../data/insurance.csv', sample_path='../data/insurance_sample.csv', n_samples=1000, random_state=42):
    """
    Load data từ file csv, nếu không có thì tạo dữ liệu mẫu và lưu lại.
    Trả về DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print("✅ Đã tải dữ liệu từ file local")
    except FileNotFoundError:
        print("⚠️ Không tìm thấy file dữ liệu. Tạo dữ liệu mẫu...")
        np.random.seed(random_state)
        data = {
            'age': np.random.randint(18, 65, n_samples),
            'sex': np.random.choice(['male', 'female'], n_samples),
            'bmi': np.random.normal(28, 6, n_samples),
            'children': np.random.randint(0, 6, n_samples),
            'smoker': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
            'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n_samples),
        }
        charges = []
        for i in range(n_samples):
            base_cost = 1000
            age_factor = data['age'][i] * 50
            bmi_factor = max(0, (data['bmi'][i] - 25) * 100)
            smoker_factor = 15000 if data['smoker'][i] == 'yes' else 0
            children_factor = data['children'][i] * 500
            total_charge = base_cost + age_factor + bmi_factor + smoker_factor + children_factor
            total_charge += np.random.normal(0, 2000)
            charges.append(max(1000, total_charge))
        data['charges'] = charges
        df = pd.DataFrame(data)
        df.to_csv(sample_path, index=False)
        print(f"✅ Đã tạo và lưu dữ liệu mẫu tại {sample_path}")
    return df

def preprocess_data(df):
    """
    Tiền xử lý dữ liệu: encode categorical, tách features/target, chia train/test.
    Trả về X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    df_processed = df.copy()
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    df_processed['sex'] = le_sex.fit_transform(df_processed['sex'])
    df_processed['smoker'] = le_smoker.fit_transform(df_processed['smoker'])
    df_processed['region'] = le_region.fit_transform(df_processed['region'])
    X = df_processed.drop('charges', axis=1)
    y = df_processed['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
