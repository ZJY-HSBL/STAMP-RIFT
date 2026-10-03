"""CLI for paper-data audits, synthetic experiments, and supplied data."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
from stamp_hf import LambdaCapacity, evaluate, normalize, distances
from stamp_hf.core import closeness, independent_topsis, compromise_baseline

ROOT = Path(__file__).resolve().parent


def save_json(path, obj):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert)+'\n', encoding='utf-8')


def paper_audit(output):
    paper = json.loads((ROOT/'data/paper_public.json').read_text())
    cap = LambdaCapacity(paper['singletons_26'])
    x = normalize(paper['matrix'])
    positive, negative, dp, dm = distances(x)
    calculated = closeness(paper['published_z_positive'], paper['published_z_negative'])
    s = np.array(paper['singletons_26'])
    forced_mass = np.expm1(np.log1p(1.2*s).sum())/1.2
    findings = {
        'scope': 'Only c1,c2,c25,c26 are public; full 26-criterion ranking is not reproducible.',
        'singleton_sum': s.sum(), 'solved_lambda': cap.lam,
        'published_lambda': 1.2, 'full_mass_at_published_lambda': forced_mass,
        'computed_positive_ideal_public_columns': positive,
        'computed_negative_ideal_public_columns': negative,
        'distance_positive_public_columns': dp,
        'distance_negative_public_columns': dm,
        'table8_closeness_recomputed_from_rounded_z': calculated,
        'table8_reported_closeness': paper['published_scores'],
        'table8_closeness_difference': calculated - paper['published_scores'],
        'table8_reported_order_only': ['a4','a2','a3','a1'],
        'findings': [
            'Eq. 13 uses max for the negative ideal; use coordinate-wise min.',
            'Eq. 2 states d(A,A)=1; the implemented RMS distance has d(A,A)=0.',
            'Printed c26 negative ideal ends in 0.9; the public data imply 0.8.',
            'Printed D+ for a2,c1 is 0.1205; the public HFE equals the positive ideal, so distance is zero.',
            'Lambda=1.2 does not normalize the published Table 6 singleton measures.',
            'General lambda capacities have higher-order interactions and are not generally 2-additive.',
            'Table 8 rounded distances and scores do not agree exactly; these are reported values, not reproduced outputs.',
            'Missing c3-c24 prevents independent reproduction of Tables 8-10.'
        ]
    }
    save_json(output/'paper_audit.json', findings)
    lines=['# 公开数据核验','', '以下数值由公开数据和公式重新计算；不代表完整案例复现。','',
           f'- 表 6 单元素测度之和：{s.sum():.10f}',
           f'- 满足归一化的 λ：{cap.lam:.10f}',
           f'- 强制 λ=1.2 时全集测度：{forced_mass:.10f}',
           '- c1 的 a2 正理想解距离：0（论文列为 0.1205）。',
           '- c26 的负理想解应为 [0.7,0.7,0.7,0.7,0.8]。','',
           '|方案|表 8 贴近度|用表 8 距离重算|差值|','|---|---:|---:|---:|']
    for i,name in enumerate(paper['alternatives']):
        lines.append(f'|{name}|{paper["published_scores"][i]:.4f}|{calculated[i]:.8f}|{calculated[i]-paper["published_scores"][i]:+.8f}|')
    lines += ['', 'c3—c24 的原始 HFE 未公开，完整排序无法独立复算。模拟实验未填入原文数据文件。']
    (output/'paper_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:findings[k] for k in ['singleton_sum','solved_lambda','full_mass_at_published_lambda']},indent=2))


def export_result(dataset, theta, output):
    matrix = dataset['matrix']
    if any(cell is None for row in matrix for cell in row):
        raise ValueError('missing HFE cells: supply actual c3-c24 observations; no imputation is performed')
    cap = LambdaCapacity(dataset['singletons'])
    result = evaluate(matrix, cap, theta, dataset.get('benefit'))
    names = dataset.get('alternatives', [f'a{i+1}' for i in range(len(matrix))])
    if len(names) != len(matrix) or len(set(names)) != len(names):
        raise ValueError('alternative labels must be unique and match matrix rows')
    result['ranking'] = [names[i] for i in result['order']]
    result['lambda'] = cap.lam
    result['theta'] = theta
    result['provenance'] = dataset.get('provenance', 'user-supplied data')
    save_json(output/'result.json', result)
    with (output/'ranking.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['position','alternative','z_positive','z_negative','closeness'])
        for rank,i in enumerate(result['order'],1):
            writer.writerow([rank,names[i],result['z_positive'][i],result['z_negative'][i],result['scores'][i]])
    print(' > '.join(result['ranking']))
    return result


def demo(output, seed):
    paper=json.loads((ROOT/'data/paper_public.json').read_text())
    rng=np.random.default_rng(seed)
    # Fully synthetic, including the four columns that are public in the paper.
    matrix=[[np.round(rng.uniform(.1,.95,size=int(rng.integers(1,6))),3).tolist() for _ in range(26)] for _ in range(4)]
    dataset=dict(provenance=f'SYNTHETIC demonstration, all 104 HFEs generated with seed={seed}; singleton measures from Table 6.',
                 alternatives=['a1','a2','a3','a4'],criteria=[f'c{i+1}' for i in range(26)],matrix=matrix,
                 singletons=paper['singletons_26'],benefit=[True]*26)
    save_json(output/'synthetic_input.json',dataset)
    result=export_result(dataset,0,output)
    cap=LambdaCapacity(dataset['singletons'])
    curve=np.array([evaluate(matrix,cap,float(t))['scores'] for t in np.linspace(0,1,21)])
    with (output/'risk_sensitivity.csv').open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['theta','a1','a2','a3','a4'])
        writer.writerows(np.column_stack([np.linspace(0,1,21),curve]))
    independent=independent_topsis(matrix,cap.singletons)
    compromise=compromise_baseline(matrix,cap.singletons)
    save_json(output/'baseline_comparison.json', dict(provenance='SYNTHETIC; not Table 10 reproduction',
              correlation=result['scores'], independent_rms=independent, compromise_Q_lower_is_better=compromise))
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for i in range(4):
        axes[0].plot(np.linspace(0,1,21),curve[:,i],label=f'a{i+1}')
    axes[0].set(xlabel='Risk coefficient theta',ylabel='Closeness',title='Synthetic risk sensitivity')
    axes[0].legend(); axes[0].grid(alpha=.2)
    pos=np.arange(4)
    axes[1].bar(pos-.18,result['scores'],.36,label='Correlation integral')
    axes[1].bar(pos+.18,independent,.36,label='Independent RMS')
    axes[1].set(xticks=pos,xticklabels=dataset['alternatives'],ylabel='Closeness',title='Synthetic baseline comparison')
    axes[1].set_ylim(0, 0.7); axes[1].legend(); fig.savefig(output/'synthetic_experiments.png',dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['audit','demo','evaluate']:
        p=sub.add_parser(command)
        p.add_argument('--output',type=Path,default=ROOT/'results'/command)
        if command=='demo': p.add_argument('--seed',type=int,default=2026)
        if command=='evaluate':
            p.add_argument('--input',type=Path,required=True)
            p.add_argument('--theta',type=float,default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        if args.command=='audit': paper_audit(args.output)
        elif args.command=='demo': demo(args.output,args.seed)
        else: export_result(json.loads(args.input.read_text(encoding='utf-8')),args.theta,args.output)
    except (ValueError,KeyError,TypeError,OSError) as error:
        parser.exit(2,f'Input or computation error: {error}\n')

if __name__=='__main__': main()
