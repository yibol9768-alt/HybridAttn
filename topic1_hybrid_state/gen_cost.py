import json
d=json.load(open('../results_r22/bench_r22_out/summary_r22.json'))
models=['0.8B','2B','4B','9B']
out={}
for m in models:
    M2=d['models'][m]['Module2_Runtime_Archives_And_Scaling']
    M3=d['models'][m]['Module3_TaskC_Valid_K4_Branching']
    r={}
    r['mem8k2048']=M2['T8192_C2048']['archive_modes']['Delta16']['extra_archive_over_kv_ratio']
    r['mib8k2048']=M2['T8192_C2048']['archive_modes']['Delta16']['extra_archive_mib']
    r['kv8k']=M2['T8192_C2048']['full_attn_kv_mib']
    for C in ['1024','2048']:
        a=M2['T8192_C'+C]['archive_modes']['Delta16']
        r['edit_sp_'+C]=a['speedup_vs_SparseAnchor']; r['edit_cp_'+C]=a['speedup_vs_Compact']
    for C in ['256','1024']:
        b=M3[C]
        r['br_sp_'+C]=b['speedup_symmetric_K4_Delta16_vs_SparseAnchor']
        r['br_cp_'+C]=b['speedup_symmetric_K4_Delta16_vs_Compact']
        r['br_post_'+C]=b['speedup_post_archive_K4_Delta32_vs_SparseAnchor']
        r['kstar_'+C]=b['K_star_Delta16_vs_SparseAnchor']
    out[m]=r
json.dump(out,open('cost_numbers.json','w'),indent=1)
for m in models: print(m,{k:(round(v,3) if isinstance(v,float) else v) for k,v in out[m].items()})
L=[]
L.append('\\begin{tabular}{lccccccc}\n\\toprule\n')
L.append(' & \\multicolumn{1}{c}{Archive} & \\multicolumn{2}{c}{Edit, 8K tokens} & \\multicolumn{4}{c}{Four branches, 4K tokens, build charged}\\\\\n')
L.append('\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\n')
L.append(' & \\multicolumn{1}{c}{vs.\\ KV} & \\multicolumn{2}{c}{vs.\\ suffix replay} & \\multicolumn{2}{c}{vs.\\ suffix replay} & \\multicolumn{2}{c}{vs.\\ re-prefill}\\\\\n')
L.append('Model & $C{=}2048$ & $C{=}1024$ & $C{=}2048$ & $C{=}256$ & $C{=}1024$ & $C{=}256$ & $C{=}1024$\\\\\n\\midrule\n')
for m in models:
    r=out[m]
    L.append(f"{m} & {r['mem8k2048']:.2f}$\\times$ & {r['edit_sp_1024']:.1f}$\\times$ & {r['edit_sp_2048']:.1f}$\\times$ & {r['br_sp_256']:.2f}$\\times$ & {r['br_sp_1024']:.2f}$\\times$ & {r['br_cp_256']:.2f}$\\times$ & {r['br_cp_1024']:.2f}$\\times$\\\\\n")
L.append('\\bottomrule\n\\end{tabular}\n')
open('tables/cost.tex','w').write(''.join(L))

# appendix branching table (Delta16)
B=[]
B.append('\\begin{tabular}{llcccc}\n\\toprule\nModel & $C$ & vs.\\ suffix replay & vs.\\ re-prefill & $K^\\star$ (8 contexts) \\\\\n\\midrule\n')
for m in models:
    for C in ['256','1024']:
        r=out[m]; ks=r['kstar_'+C]
        kk=f"{min(ks)}" if min(ks)==max(ks) else f"{min(ks)}--{max(ks)}"
        B.append(f"{m} & {C} & {r['br_sp_'+C]:.2f}$\\times$ & {r['br_cp_'+C]:.2f}$\\times$ & {kk}\\\\\n")
B.append('\\bottomrule\n\\end{tabular}\n')
open('tables/branch.tex','w').write(''.join(B))
