import json, ast
nb = json.load(open('multi_dataset_integration.ipynb', encoding='utf-8'))
errors = []
code_cells = 0
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', ''))
    if not src.strip():
        continue
    code_cells += 1
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append(f'Cell {i}: {e}')

if errors:
    for e in errors:
        print(e)
else:
    print(f'ALL {code_cells} code cells: SYNTAX OK')
