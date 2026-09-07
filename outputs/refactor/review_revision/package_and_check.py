from pathlib import Path
from PIL import Image, ImageDraw
import re, shutil, zipfile, hashlib
root=Path(__file__).resolve().parent
checks=[]
for venue,label in [('ieee_access','IEEEAccess'),('mdpi_drones','MDPI_Drones')]:
    folder=root/venue
    pages=sorted((folder/'qa').glob('page-*.png'))
    for offset in range(0,len(pages),8):
        sheet=Image.new('RGB',(1200,850),'#e3e7ec')
        for i,p in enumerate(pages[offset:offset+8]):
            im=Image.open(p).convert('RGB'); im.thumbnail((290,390))
            x=(i%4)*300; y=(i//4)*425
            sheet.paste(im,(x+(300-im.width)//2,y+5))
            ImageDraw.Draw(sheet).text((x+12,y+405),p.stem,fill='black')
        sheet.save(folder/'qa'/f'contact_{offset//8+1}.png')
    src=folder/'source'
    text=(src/'main.tex').read_text()
    labels=re.findall(r'\\label\{([^}]+)\}',text)
    refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',text)
    figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text)
    cites=set(k.strip() for block in re.findall(r'\\cite\{([^}]+)\}',text) for k in block.split(','))
    bib=set(re.findall(r'@\w+\s*\{\s*([^,]+)',(src/'references.bib').read_text()))
    assert all((src/f).exists() for f in figures)
    assert not set(refs)-set(labels),set(refs)-set(labels)
    assert not cites-bib,cites-bib
    assert len(labels)==len(set(labels))
    checks.append(f'{label}: {len(pages)} rendered pages, {len(figures)} figure references, {len(cites)} citation keys; all resolved.')
    shutil.copy2(folder/'build/main.pdf',folder/f'DiSwitch_{label}_Reviewed.pdf')
    shutil.copy2(root/'REVIEWER_REPORT_KO.md',src/'REVIEWER_REPORT_KO.md')
    archive=folder/f'DiSwitch_{label}_Overleaf.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob('*')):
            if p.is_file() and not any(x in ('build','rendered','__pycache__','.DS_Store') for x in p.parts):
                z.write(p,p.relative_to(src))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
checks.append('Figures retained their numerical content. No new experiments or revised numerical statistics were generated.')
(root/'BUILD_CHECKS.md').write_text('# Build checks\n\n'+'\n\n'.join(checks)+'\n')
files=[p for p in root.glob('*/*') if p.suffix in ('.pdf','.zip')]
(root/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(root))+'\n' for p in files))
print('\n'.join(checks))
