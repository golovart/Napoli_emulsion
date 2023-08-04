import sys
import numpy as np

if len(sys.argv)<2: folds = 1
if len(sys.argv)==2: news_bfcl = 'news_bfcl.txt'
else: news_bfcl = sys.argv[1]
folds = int(sys.argv[-1])
news = np.loadtxt(news_bfcl, delimiter=',', dtype=int)
N = news.shape[0]
for i in range(folds):
	np.savetxt('news_bfcl'+str(i+1)+'.txt',news[i*N//folds:(i+1)*N//folds],fmt='%d',delimiter=',')
