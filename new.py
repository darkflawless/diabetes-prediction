


#B22DCCN184 Đỗ Thành Đạt 

import numpy as np 
import pandas as pd 


df = pd.read_csv('diabetes.csv')
# print(df.info())


print("\nKiểm tra giá trị thiếu:")
print(df.isnull().sum())

columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_to_check:
    zero_count = (df[column] == 0).sum()
    print(f"Số lượng giá trị 0 trong cột {column}: {zero_count}")

# Thay thế giá trị 0 bằng trung bình của cột (trừ Outcome)
for column in columns_to_check:
    df[column] = df[column].replace(0, df[column].mean())

# Hiển thị dữ liệu sau khi làm sạch
print("\nDữ liệu sau khi làm sạch:")
print(df.head())

# Lưu dữ liệu đã làm sạch
df.to_csv('cleaned_diabetes.csv', index=False)
print("Dữ liệu đã được lưu vào 'cleaned_diabetes.csv'")