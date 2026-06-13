from pathlib import Path
import fitz

root=Path(__file__).resolve().parents[2]
for filename,page,label in [('Healthcare_Bias_Mitigation.pdf',7,'paper_latent_page8'),('PPT__Copy__1 (1).pdf',20,'slide_latent_page21')]:
    doc=fitz.open(root/filename)
    doc[page].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(root/'latent/code/references'/f'{label}.png')
print('Rendered relevant latent-method and metric pages')
