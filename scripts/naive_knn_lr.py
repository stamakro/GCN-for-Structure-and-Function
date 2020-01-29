import sys
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation import smin, fmax


def load_set(set_file, feats_type, feats_dir, Nterms):
	names = np.loadtxt(set_file, dtype=str).tolist()
	Y = np.zeros((len(names), Nterms))

	if feats_type == 'embeddings':
		X = np.zeros((len(names), 1024))
		for i, p in enumerate(names):
			with open(feats_dir + '/' + p + '.pkl', 'rb') as fr:
				d = pickle.load(fr)
			X[i] = np.mean(d['embeddings'], 0).reshape(-1)
			Y[i] = d['labels'].toarray().reshape(-1)

	elif feats_type == 'onehot':
		alphabet = 'ARNDCQEGHILKMFPSTWYVUOBZJX'
		ohdict = dict((c, i) for i, c in enumerate(alphabet))
		X = np.zeros((len(names), len(ohdict)))
		for i, p in enumerate(names):
			with open(feats_dir + '/' + p + '.pkl', 'rb') as fr:
				d = pickle.load(fr)
			seq = d['sequence']
			seqlen = len(seq)
			features = np.zeros((seqlen, len(ohdict)), dtype=np.float32)
			for k in range(seqlen):
				features[k, ohdict[seq[k]]] = 1
			X[i] = np.mean(features, 0)
			Y[i] = d['labels'].toarray().reshape(-1)

	return X, Y


data_dir, feats_dir, Nterms, feats_type, out_dir = sys.argv[1:]
Nterms = int(Nterms)

ks = [1, 2, 3, 5, 7, 11, 15, 21, 25] 	# for knn
Cs = [1e-5, 1e-4, 1e-3] 				# for regularization in log reg

# Data loading
termIC = np.load(data_dir + '/icVec.npy')
assert termIC.shape[0] == Nterms

Xtrain, Ytrain = load_set(data_dir + '/train.names', feats_type, feats_dir, Nterms)
Xval, Yval = load_set(data_dir + '/valid.names', feats_type, feats_dir, Nterms)
Xtest, Ytest = load_set(data_dir + '/test.names', feats_type, feats_dir, Nterms)

print(Ytrain.shape)
print(Yval.shape)
print(Ytest.shape)
ii = np.where(np.sum(Ytest, 0) > 0)[0]



# Naive
print('\nNaive')
freq = np.sum(Ytrain, 0) / Ytrain.shape[0]
Ypost_naive = np.tile(freq, (Ytest.shape[0], 1))

with open(out_dir + '/test_pred_naive.pkl', 'wb') as fw:
	pickle.dump({'y_true': Ytest, 'y_pred': Ypost_naive}, fw)

print('AvgPrec: ', average_precision_score(Ytest, Ypost_naive, average='samples'))
print('ROCAUC: ', roc_auc_score(Ytest[:, ii], Ypost_naive[:, ii], average='macro'))
print('Smin: ', smin(Ytest, Ypost_naive, termIC, 51))
print('Fmax: ', fmax(Ytest, Ypost_naive, 51))



# K-Nearest Neighbors
print('\nk-NN ', feats_type)
aucKnn = np.zeros((len(ks),))
usedClfs = []
for i, k in enumerate(ks):
	clf = KNeighborsClassifier(n_neighbors = k)
	clf.fit(Xtrain, Ytrain)
	usedClfs.append(clf)

	Ypost_t = clf.predict_proba(Xval)
	Ypost = np.zeros(Yval.shape)

	for j, y in enumerate(Ypost_t):
		Ypost[:, j] = y[:, 1]

	aucKnn[i] = np.nanmean(roc_auc_score(Yval, Ypost, average=None))
	print(aucKnn[i])

bestK = ks[np.argmax(aucKnn)]
print('k-NN best K: %d' % bestK)
clf = usedClfs[np.argmax(aucKnn)]

Ypost_t = clf.predict_proba(Xtest)
Ypost_knn = np.zeros(Ytest.shape)

for j, y in enumerate(Ypost_t):
	Ypost_knn[:, j] = y[:, 1]

with open(out_dir + '/test_pred_knn_%s.pkl' % feats_type, 'wb') as fw:
	pickle.dump({'y_true': Ytest, 'y_pred': Ypost_knn}, fw)

print('AvgPrec: ', average_precision_score(Ytest, Ypost_knn, average='samples'))
print('ROCAUC: ', roc_auc_score(Ytest[:, ii], Ypost_knn[:, ii], average='macro'))
print('Smin: ', smin(Ytest, Ypost_knn, termIC, 51))
print('Fmax: ', fmax(Ytest, Ypost_knn, 51))



# Logistic Regression
print('\nLogistic Regression ', feats_type)
aucLin = np.zeros((len(Cs),))
usedClfs = []
for i, C in enumerate(Cs):
	print(i)
	clf = MultiOutputClassifier(SGDClassifier(loss='log', penalty='l2', alpha=C))
	clf.fit(Xtrain, Ytrain)
	usedClfs.append(clf)

	Ypost_t = clf.predict_proba(Xval)
	Ypost = np.zeros(Yval.shape)

	for j, y in enumerate(Ypost_t):
		Ypost[:, j] = y[:, 1]

	aucLin[i] = np.nanmean(roc_auc_score(Yval, Ypost, average=None))
	print(aucLin[i])

bestC = Cs[np.argmax(aucLin)]
print('LR best C: %f' % bestC)
clf = usedClfs[np.argmax(aucLin)]

Ypost_t = clf.predict_proba(Xtest)
Ypost_lin = np.zeros(Ytest.shape)

for j, y in enumerate(Ypost_t):
	Ypost_lin[:, j] = y[:, 1]

with open(out_dir + '/test_pred_logit_%s.pkl' % feats_type, 'wb') as fw:
	pickle.dump({'y_true': Ytest, 'y_pred': Ypost_lin}, fw)

print('AvgPrec: ', average_precision_score(Ytest, Ypost_lin, average='samples'))
print('ROCAUC: ', roc_auc_score(Ytest[:, ii], Ypost_lin[:, ii], average='macro'))
print('Smin: ', smin(Ytest, Ypost_lin, termIC, 51))
print('Fmax: ', fmax(Ytest, Ypost_lin, 51))
