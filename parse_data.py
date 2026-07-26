import json, re

with open(r'c:\Users\HELLO\Downloads\kid2\ckd_eda_dashboard.html', 'r', encoding='utf-8') as f:
    text = f.read()

match = re.search(r'<script id="dashdata" type="application/json">(.*?)</script>', text, re.DOTALL)
data = json.loads(match.group(1))

print("Features:")
for feat in data['features']:
    print(f"  {feat['key']} - {feat['label']}")

print(f"\ncorr_labels: {data.get('corr_labels')}")
print(f"all_features: {data.get('all_features')}")
print(f"dataset_info: {data.get('dataset_info')}")
print(f"all_labels keys: {list(data.get('all_labels', {}).keys())}")
