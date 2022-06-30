#!/usr/bin/env python2
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traintxt', type=str, default='./list/AIMT15_train.list')
    parser.add_argument('--testtxt', type=str, default='./list/AIMT15_test.list')
    parser.add_argument('--respath', type=str, default='/home/yi/me15res')
    parser.add_argument('--outpath', type=str, default='./out/aimt15_tsn,emo,mkt')
    parser.add_argument('--feas', type=str, default='tsn,emo,mkt')
    args = parser.parse_args()
    
    pres = args.respath
    lpath = {'mkt':'%s/mktdes'%pres,'tsn':['%s/me15tsn_rgb'%pres,'%s/me15tsn_flow'%pres],
        'dsift':'%s/dsiftdes'%pres,'hsh':'%s/hshdes'%pres,'emo':'%s/emocsv'%pres,
        'mfcc':'%s/mfccdes'%pres,'emolarge':'%s/emolarge'%pres,'is13':'%s/emoIS13'%pres}
    lfeas = args.feas.split(',')

    testid,testval,testaro = affutil.readlist3(args.testtxt)
    vtest,_,_ = affutil.loadfeas(testid,lfeas,lpath,pres,'test','AIMT15')

    sys.stdout.flush()

    #valence
    c = -3
    m = svm.load_model('./models/AIMT15_Val_%d.model'%(c))
    vallab, valp, valv = svm.predict(testval, vtest, m)
    print("ACC in valence=%f\n"%(valp[0]))


    #arousal
    c = -1
    m = svm.load_model('./models/AIMT15_Aro_%d.model'%(c))
    arolab, arop, arov= svm.predict(testaro, vtest, m)
    print("ACC in arousal=%f\n"%(arop[0]))
                
