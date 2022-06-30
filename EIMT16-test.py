#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:35:30 2017

@author: yi
"""
import argparse
import sys
import numpy as np
sys.path.append('./liblinear-multicore-2.11-2/python')
sys.path.append('./liblinear-multicore-2.11-2')
import liblinearutil as svm
import affutil
import eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traintxt', type=str, default='./list/EIMT16_train.list')
    parser.add_argument('--testtxt', type=str, default='./list/EIMT16_test.list')
    parser.add_argument('--respath', type=str, default='./data')
    parser.add_argument('--outpath', type=str, default='./out/me16_tsn,emo,mkt')
    parser.add_argument('--feas', type=str, default='tsn,emo,mkt')
    args = parser.parse_args()
  
    pres = args.respath
    lpath = {'mkt':'%s/mktdes'%pres,'tsn':['%s/me16tsn_rgb'%pres,'%s/me16tsn_flow'%pres],
        'dsift':'%s/dsiftdes'%pres,'hsh':'%s/hshdes'%pres,'emo':'%s/emocsv'%pres,
        'mfcc':'%s/mfccdes'%pres,'emolarge':'%s/emolarge'%pres,'is13':'%s/emoIS13'%pres}
    lfeas = args.feas.split(',')

    testid,testval,testaro = affutil.readlist3(args.testtxt)
    vtest,vmean,vstd = affutil.loadfeas(testid,lfeas,lpath,pres,'test','EIMT16')

    sys.stdout.flush()
               
    ntest = len(testid)

    #valence
    c = -5
    m = svm.load_model('./models/EIMT16_Val_%d.model'%(c))
    vallab, valp, valv = svm.predict(testval, vtest, m)
    #arousal
    c = -2
    m = svm.load_model('./models/EIMT16_Aro_%d.model'%(c))
    arolab, arop, arov= svm.predict(testaro, vtest, m)
    mseval, pccval, msearo, pccaro = eval.evalEIMT16(valv,arov,'./list/MEDIAEVAL16-Global_prediction.txt')

    print('MSE in valence=%f\t PCC in valence=%f\nMSE in arousal=%f\t PCC in arousal=%f\t\n'
          %(mseval, pccval, msearo, pccaro))
    with open('%s_%d.txt'%(args.outpath,c),'w') as wf:
        for v in range(ntest):
            wf.write('%s %f %f\n'%(testid[v],vallab[v],arolab[v]))
