import numpy
import numpy as np
from scipy.io import loadmat

def peass_mos(final_pesq):
    
    x = loadmat('PEASS-subjdata.mat')
    scores = x['scores']
    soundnames=x['soundNames']
    scores=np.mean(scores,axis=0)
    final_mos=scores
    
    from scipy.stats import pearsonr
    corr1, _ = pearsonr(final_mos[0*80:0*80+80],final_pesq)
    from scipy.stats import spearmanr
    corr11,_=spearmanr(final_mos[0*80:0*80+80],final_pesq)
    print(corr1)
    print(corr11)
    
    from scipy.stats import pearsonr
    corr2, _ = pearsonr(final_mos[1*80:1*80+80],final_pesq)
    from scipy.stats import spearmanr
    corr22,_=spearmanr(final_mos[1*80:1*80+80],final_pesq)
    print(corr2)
    print(corr22)
    
    from scipy.stats import pearsonr
    corr3, _ = pearsonr(final_mos[2*80:2*80+80],final_pesq)
    from scipy.stats import spearmanr
    corr33,_=spearmanr(final_mos[2*80:2*80+80],final_pesq)
    print(corr3)
    print(corr33)
    
    from scipy.stats import pearsonr
    corr4, _ = pearsonr(final_mos[3*80:3*80+80],final_pesq)
    from scipy.stats import spearmanr
    corr44,_=spearmanr(final_mos[3*80:3*80+80],final_pesq)
    print(corr4)
    print(corr44)
    
    return [corr1,corr11,corr2,corr22,corr3,corr33,corr4,corr44]
    
    