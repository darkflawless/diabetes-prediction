import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
import numpy as np

# Tải dữ liệu đã làm sạch
df = pd.read_csv('cleaned_diabetes.csv')

# Chọn 5 đặc trưng và target
features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Pregnancies']
X = df[features]
y = df['Outcome']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện 5 mô hình
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Lưu kết quả MSE
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results.append([name, mse])

# In kết quả dưới dạng bảng
headers = ["Model", "MSE"]
print(tabulate(results, headers=headers, tablefmt="grid", floatfmt=".4f"))