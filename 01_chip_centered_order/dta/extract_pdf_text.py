import sys
import os

pdf_path = os.path.join('文章', 'C247-ASPDAC2025-PCBAgent.pdf')
try:
    import PyPDF2
except Exception:
    print('PyPDF2 not installed; please pip install PyPDF2')
    sys.exit(1)

def extract_text(path):
    out = []
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ''
            except Exception:
                txt = ''
            out.append(txt)
    return '\n'.join(out)

if __name__ == '__main__':
    if not os.path.exists(pdf_path):
        print('PDF not found at', pdf_path)
        sys.exit(1)
    text = extract_text(pdf_path)
    keywords = ['routing', 'route', 'wire', '布线', '走线', '连线', 'router']
    hits = []
    lower = text.lower()
    for kw in keywords:
        if kw in lower:
            hits.append(kw)
    print('keywords_found:', hits)
    # print some context for each hit
    for kw in hits:
        idx = lower.find(kw)
        start = max(0, idx-200)
        end = min(len(text), idx+200)
        print('\n--- context for', kw, '---')
        print(text[start:end])
