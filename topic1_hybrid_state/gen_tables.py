import json
d=json.load(open('../results_r2/bench_r2_out/summary_all.json'))
models=['0.8B','2B','2B-Base','4B','9B']
def g(m,C,meth,k):
    return d['models'][m]['by_C'][C]['methods_heldout_test'][meth][k]
def f(v,kind):
    if kind=='pct': return f"{100*v:.0f}"
    return f"{v:.1e}" if v<0.01 else f"{v:.3f}"
rows=[('KV-Only-Evict','KV-only evict','256'),('CLM-NearestAnchorStaleGDN','Stale-anchor (CLM)','256'),('CHymERA-F','\\chym{}-$F$ (2-pass)','256'),('CHymERA-Fast','\\chym{}-Fast','256'),('CHymERA-Fast','\\chym{}-Fast','1024')]
out=[]
out.append('\\begin{tabular}{llrrrrr}\n\\toprule\n')
out.append('Method & $C$ & ' + ' & '.join(models) + ' \\\\\n\\midrule\n')
for blk,key,title in [('KL','KL_gapped','KL to exact answer $\\downarrow$'),('EM','EM_gold','Gold exact match (\\%) $\\uparrow$')]:
    out.append('\\multicolumn{7}{l}{\\emph{%s}}\\\\\n'%title)
    for meth,name,C in rows:
        vals=[g(m,C,meth,key) for m in models]
        if blk=='KL': cells=[f(v,'kl') for v in vals]
        else: cells=[('--' if m=='2B-Base' else f(v,'pct')) for v,m in zip(vals,models)]
        if meth=='CHymERA-Fast' and C=='256': cells=['\\textbf{%s}'%c for c in cells]
        out.append(f"{name} & {C} & "+' & '.join(cells)+' \\\\\n')
    if blk!='EM': out.append('\\midrule\n')
out.append('\\bottomrule\n\\end{tabular}\n')
open('tables/main_fid.tex','w').write(''.join(out))
print(''.join(out))
