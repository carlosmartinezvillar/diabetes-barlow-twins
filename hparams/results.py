import csv
import json
import math
import os


def to_float(v):
    if v in ('True', 'False'):  # alpha_edge is logged as a bool
        return float(v == 'True')
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v


def read_log(path):
    '''
    Rows of epochs_<id>.tsv as dicts of floats. [] if the run never started.
    '''
    if not os.path.exists(path):
        return []
    with open(path) as fp:
        return [{k: to_float(v) for k, v in row.items()} for row in csv.DictReader(fp, delimiter='\t')]


def collect_results(params='./hparams/hpo_1.json', log_dir='./logs', metric='r2', out='./hparams/results_1.csv'):
    '''
    One row per experiment in params (logs written by train.py): its hyperparameters, how far it got,
    the probe scores at its last and best eval epoch. Written to out (CSV) and summarized on stdout.
    metric must be higher-is-better (r2 or spearman).
    '''
    with open(params) as fp:
        hps = [json.loads(l) for l in fp if l.strip()]

    results = []
    for hp in hps:
        rows  = read_log(f"{log_dir}/epochs_{hp['id']:03}.tsv")
        evals = [r for r in rows if not math.isnan(r.get(metric, math.nan))]

        res = {k: hp.get(k) for k in ('id', 'proj_dim', 'init_chan', 'cnn_layers', 'batch',
                                      'base_lrate', 'lrate', 'decay', 'lambda_d', 'bt_lambda')}

        # HOW FAR IT GOT
        done = int(rows[-1]['epoch']) + 1 if rows else 0
        res['epochs_done'] = done
        if done == 0:
            res['status'] = 'not started'
        elif done >= hp['epochs']:
            res['status'] = 'finished'
        else:
            res['status'] = f'stopped@{done}'  # crashed or still running

        # BEST AND LAST EVAL
        best = max(evals, key=lambda r: r[metric]) if evals else {}
        res[f'best_{metric}'] = best.get(metric, math.nan)
        res['best_epoch'] = int(best['epoch']) if best else math.nan

        last = evals[-1] if evals else {}
        res['last_eval_epoch'] = int(last['epoch']) if last else math.nan
        for k in dict.fromkeys(('r2', 'spearman', 'rmse', metric)):
            res[f'last_{k}'] = last.get(k, math.nan)

        # SANITY FLAGS
        res['alpha_edge_any'] = any(r.get('alpha_edge') == 1.0 for r in evals)  # ridge alpha hit grid edge -> widen alphas
        res['final_tloss'] = rows[-1].get('tloss', math.nan) if rows else math.nan

        results.append(res)

    # RANK BY BEST METRIC (runs without evals last)
    results.sort(key=lambda r: -math.inf if math.isnan(r[f'best_{metric}']) else r[f'best_{metric}'], reverse=True)

    # WRITE
    with open(out, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow({k: (f'{v:.5g}' if isinstance(v, float) and not math.isnan(v) else
                                 '' if isinstance(v, float) else v) for k, v in r.items()})

    summarize(results, metric)
    print(f'\nWrote {len(results)} rows to {out}')
    return results


def summarize(results, metric='r2'):

    # STATUS COUNTS
    counts = {}
    for r in results:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print('STATUS:', ', '.join(f'{k}: {v}' for k, v in sorted(counts.items())))

    fmt = lambda v: f'{v:.4f}' if isinstance(v, float) and not math.isnan(v) else '-'

    # BEST RUN PER ARCHITECTURE
    print(f'\nBEST PER ARCHITECTURE (best {metric}):')
    seen = set()
    for r in results:
        arch = (r['proj_dim'], r['init_chan'])
        if arch in seen or math.isnan(r[f'best_{metric}']):
            continue
        seen.add(arch)
        print(f"  proj_dim {r['proj_dim']:>4} init_chan {r['init_chan']:>2} | id {r['id']:03} "
              f"| {metric} {fmt(r['best_' + metric])} @ep{r['best_epoch']} | {r['status']} "
              f"| lrate {r['lrate']} decay {r['decay']} lambda_d {r['lambda_d']}")

    # TOP 10 OVERALL
    print(f'\nTOP 10:')
    for r in results[:10]:
        flag = ' (alpha at edge)' if r['alpha_edge_any'] else ''
        print(f"  id {r['id']:03} | {r['status']:<12} | best {metric} {fmt(r['best_' + metric])} @ep{r['best_epoch']} "
              f"| last {fmt(r['last_' + metric])} @ep{r['last_eval_epoch']} | proj_dim {r['proj_dim']} init_chan {r['init_chan']}{flag}")


if __name__ == '__main__':
    collect_results()
