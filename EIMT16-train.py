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
    parser.add_argument('--respath', type=str, default='/home/yi/me16res')
    parser.add_argument('--feas', type=str, default='tsn,emo,mkt')
    args = parser.parse_args()
  
    pres = args.respath
    lpath = {'mkt':'%s/mktdes'%pres,'tsn':['%s/me16tsn_rgb'%pres,'%s/me16tsn_flow'%pres],
        'dsift':'%s/dsiftdes'%pres,'hsh':'%s/hshdes'%pres,'emo':'%s/emocsv'%pres,
        'mfcc':'%s/mfccdes'%pres,'emolarge':'%s/emolarge'%pres,'is13':'%s/emoIS13'%pres}
    lfeas = args.feas.split(',')

    trainid,trainval,trainaro = affutil.readlist3(args.traintxt)
    testid,testval,testaro = affutil.readlist3(args.testtxt)
         
    vtrain,vmean,vstd = affutil.loadfeas(trainid,lfeas,lpath,pres,'train','EIMT16')
    vtest,vmean,vstd = affutil.loadfeas(testid,lfeas,lpath,pres,'test','EIMT16')

    print(vtrain[0].shape,vtest[0].shape) 
    sys.stdout.flush()
               
    c = -5
    #valence
    m = svm.train(trainval, vtrain, '-n 12 -s 11 -c %f'%(2**c))
    vallab, valp, valv = svm.predict(testval, vtest, m)
    svm.save_model('./models2/EIMT16_Val_%d.model'%(c), m)
    c = -2
    #arousal
    m = svm.train(trainaro, vtrain, '-n 12 -s 11 -c %f'%(2**c))
    arolab, arop, arov= svm.predict(testaro, vtest, m)
    mseval, pccval, msearo, pccaro = eval.evalEIMT16(valv,arov,'./list/MEDIAEVAL16-Global_prediction.txt')
    svm.save_model('./models2/EIMT16_Aro_%d.model'%(c), m)

    print('MSE in valence=%f\t PCC in valence%f\nMSE in arousal=%f\t PCC in arousal=%f\t\n'
          %(mseval, pccval, msearo, pccaro))
