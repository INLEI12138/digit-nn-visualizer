"""Quantize trained weights to int8 and embed them into the page.
Run after train.py:  python build.py  ->  ../index.html"""
import numpy as np, base64, json, os
here = os.path.dirname(os.path.abspath(__file__))
w = np.load(os.path.join(here, 'weights.npz'))
W = [w['arr_%d' % i] for i in range(3)]; B = [w['arr_%d' % i] for i in range(3, 6)]
out = {'W': [], 'B': [b.round(5).tolist() for b in B], 'shapes': [list(x.shape) for x in W]}
for x in W:
    s = float(np.abs(x).max() / 127)
    q = np.clip(np.round(x / s), -127, 127).astype(np.int8)
    out['W'].append({'s': s, 'd': base64.b64encode(q.tobytes()).decode()})
t = open(os.path.join(here, 'template.html'), encoding='utf-8').read()
open(os.path.join(here, '..', 'index.html'), 'w', encoding='utf-8').write(t.replace('__MODEL__', json.dumps(out)))
print('wrote index.html')
