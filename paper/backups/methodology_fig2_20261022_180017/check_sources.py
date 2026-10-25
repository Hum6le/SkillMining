from pathlib import Path
import re, json
from collections import Counter
root=Path(r'D:\paper\Skill_Baseline\paper\overleaf_sync')
out=Path(r'D:\paper\Skill_Baseline\paper\backups\methodology_fig2_20261022_180017')
seen=[]
def uncomment(text):
    return '\n'.join(re.split(r'(?<!\\)%',line,1)[0] for line in text.splitlines())
def read_tree(path):
    assert path.is_file(), str(path)
    if path in seen: return
    seen.append(path)
    for rel in re.findall(r'\\input\{([^}]+)\}',uncomment(path.read_text(encoding='utf-8'))):
        target=root/rel
        if not target.suffix: target=target.with_suffix('.tex')
        read_tree(target)
read_tree(root/'main.tex')
texts={p:uncomment(p.read_text(encoding='utf-8')) for p in seen}
alltext='\n'.join(texts.values())
labels=re.findall(r'\\label\{([^}]+)\}',alltext)
duplicates=[k for k,n in Counter(labels).items() if n>1]
refs=re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}',alltext)
undefined=sorted(set(refs)-set(labels))
assert not duplicates, duplicates
assert not undefined, undefined
figures=[]
for p,text in texts.items():
    stack=[]
    for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',text):
        if kind=='begin': stack.append(name)
        else:
            assert stack and stack.pop()==name,(str(p),kind,name)
    assert not stack,(str(p),stack)
    depth=0
    for brace in re.findall(r'(?<!\\)[{}]',text):
        depth += 1 if brace=='{' else -1
        assert depth>=0,str(p)
    assert depth==0,(str(p),depth)
    for image in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
        assert (root/image).is_file(),image
        figures.append(image)
bib=(root/'references.bib').read_text(encoding='utf-8')
bibkeys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib))
cites=set()
for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}',alltext):
    cites.update(k.strip() for k in group.split(','))
assert not cites-bibkeys,sorted(cites-bibkeys)
method=texts[root/'sections/3_methodology.tex']
subsections=re.findall(r'\\subsection\{([^}]+)\}',method)
assert len(subsections)==4,subsections
assert 'shortlist' not in method
assert 'W_{\\mathrm{tree}}' not in method
assert r'\paragraph{Evidence induction and skill compilation.}' in method
assert method.count(r'\label{fig:skillminer-framework}')==1
assert r'\begin{figure*}[t]' in method
assert not re.search(r'^%%', (root/'sections/3_methodology.tex').read_text(encoding='utf-8'),re.M)
algorithm=re.search(r'\\begin\{algorithmic\}.*?\\end\{algorithmic\}',method,re.S).group()
stack=[]
for token in re.findall(r'\\(ForAll|For|EndFor|If|EndIf)\b',algorithm):
    if token in ['For','ForAll']: stack.append('For')
    elif token=='If': stack.append('If')
    elif token=='EndFor': assert stack.pop()=='For'
    elif token=='EndIf': assert stack.pop()=='If'
assert not stack
report={'active_tex_files':[str(p.relative_to(root)) for p in seen],
        'subsections':subsections,'figure_files':figures,
        'figure_number_2_expected':figures[1]=='figures/WWW 27 fig 2.pdf',
        'undefined_references':undefined,'duplicate_labels':duplicates,
        'cited_bib_entries':len(cites),'braces_and_environments':'passed',
        'algorithm_nesting':'passed','local_tex_compiler':'not available'}
(out/'source_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))

