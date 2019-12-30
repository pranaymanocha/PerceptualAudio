import numpy
import numpy as np


def voice_conversion_mos(final_pesq,args):
    
    MOS_HUB={
    'N10':4.06,
    'B01':3.57,
    'N17':3.41, 
    'N08':3.35,
    'D02':3.27,
    'N12':3.21,
    'N13':3.21,
    'N04':3.21,
    'N15':3.02,
    'D04':2.94,
    'N11':2.93,
    'N05':2.90,
    'N18':2.89,
    'N20':2.85,
    'D05':2.80,
    'N09':2.76,
    'D03':2.76,
    'D01':2.68,
    'N14':2.68,
    'N07':2.60,
    'N03':2.53,
    'N16':2.07,
    'N06':2.00,
    'N19':1.96}

    MOS_SPO={
    'N10':4.12,
    'N13':2.99,
    'B01':2.94,
    'N17':2.89,
    'N18':2.84,
    'N12':2.83,
    'N11':2.83,
    'N04':2.78,
    'N05':2.74,
    'N03':2.53,
    'N06':2.32,
    'N16':1.92}
    
    final_mos=[]
    
    if args.HUB_SPO==0:
        sorting=sorted(MOS_HUB)
        for sorteded in sorting:
            final_mos.append(MOS_HUB[sorteded])
    elif args.HUB_SPO==1:
        sorting=sorted(MOS_SPO)
        for sorteded in sorting:
            final_mos.append(MOS_SPO[sorteded])
            
    from scipy.stats import pearsonr
    corr, _ = pearsonr(final_mos,final_pesq)
    from scipy.stats import spearmanr
    corr1,_=spearmanr(final_mos,final_pesq)
    print(corr)
    print(corr1)
    
    return [corr,corr1]