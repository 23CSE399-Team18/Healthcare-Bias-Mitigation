from pathlib import Path
import importlib.util
import json
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / '_pdf_reader'))
from pypdf import PdfReader
out = root / 'latent/work'
out.mkdir(parents=True, exist_ok=True)
for p in root.glob('*.pdf'):
    reader = PdfReader(p)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ''
        if any(word in text.lower() for word in ('latent', 'feature extraction', 'autoencoder', 'representation learning')):
            pages.append({'page': i+1, 'text': text})
    (out / (p.stem+'.json')).write_text(json.dumps(pages, indent=2), encoding='utf-8')
    print(json.dumps({'pdf': p.name, 'pages': pages}, ensure_ascii=True))
availability = {n: importlib.util.find_spec(n) is not None for n in ['torch','sklearn','matplotlib','scipy','nbformat','pymupdf','fitz']}
print(json.dumps({'runtime': sys.executable, 'packages': availability}))
if availability['torch']:
    import torch
    print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available()}))
