import json
import numpy as np
import random


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def sample(spec):
    '''
    ('log',lo,hi)   -> log-uniform in [lo,hi]
    ('choice',[..]) -> uniform over the list
    '''
    if spec[0] == 'log':
        _, lo, hi = spec
        return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
    return spec[1][np.random.randint(len(spec[1]))]


def set_experiments_1(n=30, epochs=110, seed=476, path='./hparams/hpo_1.json'):
    '''
    n random optimizer/loss samples for each architecture (proj_dim x init_chan), fixed batch.
    lrate = base_lrate * sqrt(batch/128), so base_lrate stays comparable if batch is changed later.
    '''
    set_seed(seed)

    space = {
        'base_lrate': ('log', 1e-4, 2e-3),       # lr at batch 128
        'decay':      ('log', 1e-4, 5e-2),
        'lambda_d':   ('log', 1.0, 40.0),        # bt_lambda * proj_dim (off-diag terms grow ~D^2)
    }


    # FIXED
    batch = 128
    cnn_layers = 3
    encoder_type = 'cnn'


    # ARCHITECTURES (representation dim = 32*init_chan)
    grid = []
    for proj_dim in [1024, 2048, 4096]:
        for init_chan in [32, 48]:
            grid.append((proj_dim, init_chan))

    i = 0
    with open(path, 'w') as fp:
        for proj_dim, init_chan in grid:
            for _ in range(n):

                exp = {k: sample(v) for k, v in space.items()}
                exp['proj_dim']   = proj_dim
                exp['init_chan']  = init_chan
                exp['cnn_layers'] = cnn_layers
                exp['batch']      = batch  # images per batch
                exp['lrate']      = exp['base_lrate'] * (batch / 128) ** 0.5 #lrate adjust rule
                exp['encoder']    = encoder_type

                # ROUND (3 SIGNIFICANT FIGURES)
                for k in ('base_lrate', 'lrate', 'decay', 'lambda_d'):
                    exp[k] = float(f"{exp[k]:.3g}")
                exp['bt_lambda'] = float(f"{exp['lambda_d'] / proj_dim:.3g}")

                exp.update({'id': i, 'seed': seed, 'epochs': epochs, 'bands': 10})
                fp.write(json.dumps(exp) + '\n')
                i += 1


if __name__ == '__main__':
    set_experiments_1(n=30,epochs=110,seed=476,path='./hparams/hpo_1.json')
