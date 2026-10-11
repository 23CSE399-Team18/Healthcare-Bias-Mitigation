import pandas as pd

csv_path = r"e:\Data_Set\integrated_output\full_integrated_dataset.csv"
df_head = pd.read_csv(csv_path, nrows=5)
columns = list(df_head.columns)

cdc_non_null = ['dataset_source', 'domain', 'target', 'age', 'age_group', 'gender', 'BMI', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'age_code', 'Education', 'Income', 'age_label']

print("Column name indices:")
for col in cdc_non_null:
    idx = columns.index(col)
    print(f"  - {col}: column index {idx} (Excel Column {chr(65 + idx) if idx < 26 else chr(65 + (idx // 26) - 1) + chr(65 + (idx % 26))})")
