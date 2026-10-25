"""Render every final report page for visual inspection using the bundled runtime."""
from pathlib import Path
import pypdfium2 as pdfium

root=Path(__file__).resolve().parents[1]
source=root/'outputs/report/MAAI901_Technical_Report_Ayman_Lakhnati.pdf'
destination=root/'build/report'
destination.mkdir(parents=True,exist_ok=True)
document=pdfium.PdfDocument(source)
for index in range(len(document)):
    page=document[index]
    bitmap=page.render(scale=1.5)
    bitmap.to_pil().save(destination/f'page-{index+1:02d}.png')
    bitmap.close()
    page.close()
document.close()
print(destination)
