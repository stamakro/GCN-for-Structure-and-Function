import sys
import pickle
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from evaluation import smin, fmax, bootstrap


filename, icvecfile, model, nboots = sys.argv[1:]


with open(filename, 'rb') as fr:
    d = pickle.load(fr)
    
Ytest = d['y_true']
Ypost = d['y_pred']

termIC = np.load(icvec)

print(model)
print('AvgPrec: ', average_precision_score(Ytest, Ypost, average='samples'))
print('ROCAUC: ', roc_auc_score(Ytest, Ypost, average='macro'))
print('Smin: ', smin(Ytest, Ypost, termIC, 51))
print('Fmax: ', fmax(Ytest, Ypost, 51))

boot = bootstrap(Ytest, Ypost, termIC, nrBootstraps=int(nboots), nrThresholds=51)

auc_p_bootstraps = boot['auc']
roc_t_bootstraps = boot['roc']
sd_min_bootstraps = boot['sd']
fmax_bootstraps = boot['fmax']

print('AvgPrec CI: ', np.percentile(auc_p_bootstraps, [2.5, 97.5]))
print('ROCAUC CI: ', np.percentile(roc_t_bootstraps, [2.5, 97.5]))
print('Smin CI: ', np.percentile(sd_min_bootstraps, [2.5, 97.5]))
print('Fmax CI: ', np.percentile(fmax_bootstraps, [2.5, 97.5]))
