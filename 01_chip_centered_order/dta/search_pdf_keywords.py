import os
import sys
try:
    import PyPDF2
except Exception:
    print('PyPDF2 not installed; please pip install PyPDF2')
    sys.exit(1)

pdf_path = os.path.join('文章', 'C247-ASPDAC2025-PCBAgent.pdf')
if not os.path.exists(pdf_path):
    print('PDF not found:', pdf_path)
    sys.exit(1)

keywords = [
    'action', 'action space', 'action-space', 'place', 'placement', 'place(', 'place ',
    'route', 'routing', 'router', 'via', 'layer', 'layers', 'detailed routing', 'global routing',
    'joint', 'jointly', 'end-to-end', 'reinforcement', 'rl', 'policy', 'agent', 'step', 'move'
]

def extract_pages(path):
    texts = []
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ''
            except Exception:
                txt = ''
            texts.append(txt)
    return texts

def search(texts):
    hits = []
    for i, t in enumerate(texts):
        low = t.lower()
        for kw in keywords:
            if kw in low:
                idx = low.find(kw)
                start = max(0, idx-200)
                end = min(len(t), idx+200)
                context = t[start:end].replace('\n',' ')
                hits.append((i+1, kw, context))
    return hits

if __name__ == '__main__':
    pages = extract_pages(pdf_path)
    results = search(pages)
    if not results:
        print('No keyword hits')
        sys.exit(0)
    # Group by page
    from collections import defaultdict
    bypage = defaultdict(list)
    for p, kw, c in results:
        bypage[p].append((kw, c))
    for p in sorted(bypage.keys()):
        print('\n=== PAGE', p, '===')
        seen = set()
        for kw, c in bypage[p]:
            if (kw, c[:100]) in seen:
                continue
            seen.add((kw, c[:100]))
            print('\n[kw]', kw)
            print(c)
    # Save pages that have routing keywords to output file for implementation reference
    out_lines = []
    for p in sorted(bypage.keys()):
        out_lines.append('\n=== PAGE %d ===\n' % p)
        out_lines.append(pages[p-1])
    os.makedirs('output', exist_ok=True)
    with open('output/paper_pages.txt', 'w', encoding='utf-8') as fo:
        fo.write('\n'.join(out_lines))
    print('\nSaved matched pages text to output/paper_pages.txt')
