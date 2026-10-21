import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent/'hybrid'
h=pd.DataFrame(json.loads((out/'training-history.json').read_text()))
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
train=h.dropna(subset=['training_loss'])
axes[0].plot(train.epoch,train.training_loss,'o-',color='#16756d')
axes[0].set(title='Global hybrid training',xlabel='Training round',ylabel='Reconstruction objective')
dev=h.dropna(subset=['development_score'])
axes[1].plot(dev.epoch,dev.development_score,'o-',color='#315d9e')
axes[1].axhline(dev.development_score.iloc[0],color='#999',ls='--',label='Warm-start model')
axes[1].set(title='Internal development only',xlabel='Training round',ylabel='Task 1 score')
axes[1].legend(fontsize=8)
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.savefig(out/'live-progress.png',dpi=150)
print('Hybrid learning curve saved.')
