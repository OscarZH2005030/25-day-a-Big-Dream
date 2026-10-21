import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

root=Path(__file__).resolve().parent
h=pd.read_json(root/'training/training-history.json')
d=h.dropna(subset=['development_score'])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),layout='constrained')
fig.suptitle('10.86M-parameter graph state-space model | Training in progress',fontsize=14)
axes[0].plot(h.epoch,h.training_loss,'o-',color='#16756d')
axes[0].set(xlabel='Completed round',ylabel='Reconstruction objective',title='Training')
axes[1].plot(d.epoch,d.development_score,'o-',color='#315d9e',label='EMA graph-SSM')
axes[1].axhline(d.initial_score.iloc[0],color='#888',ls='--',label='Initial reconstruction')
axes[1].set(xlabel='Completed round',ylabel='Task 1 score',title='Internal development: Dec 14 - Jan 4')
axes[1].legend(fontsize=8)
last=d.iloc[-1]
axes[1].annotate(f'{last.development_score:.5f}',(last.epoch,last.development_score),xytext=(-50,-18),textcoords='offset points',fontsize=9)
for ax in axes:ax.grid(alpha=.18)
fig.savefig(root/'training/live-learning-curve.png',dpi=145)
print('Learning curve updated.')
