#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:35:30 2017

@author: yi
"""
import numpy as np
import cv2
import sys
import scipy.io as sio
import os

def norm1dsignedl2(dat):
    assert(dat.shape[0]==1 or len(dat.shape)==1)
    dt = np.sign(dat)*np.sqrt(np.abs(dat))
    norm = np.sqrt(np.sum(dt*dt)) + 1e-30
    dt = dt/norm         
    return dt

def readcsv(csvpath):
    with open(csvpath) as rf:
        lines = rf.readlines()
        fea = lines[-1].split(',')[1:-1]
    return np.array([np.float(x) for x in fea])

def readlist3(ltxt):
    vvid = []
    val = []
    aro = []
    with open(ltxt,'r') as fo:
        lines = fo.readlines()
        for line in lines:
            spl = line.split(' ')
            if len(spl) == 3:
                vvid.append(spl[0])
                val.append(float(spl[1]))
                aro.append(float(spl[2]))
            else:
                print('[ERROR] len(spl) == 2',line)
                exit(0)
    return vvid, np.array(val), np.array(aro)

def readonemkt(pmkt,vid):
    tfea = []
    for f in ['THOG','THOF','TMBHX','TMBHY','TBCF']:
        ftmp = '%s/%s/%s.gz'%(pmkt,f,vid)
        fs = cv2.FileStorage(ftmp,0)
        dtmp = fs.getNode('vocabulary').mat()
        tfea.append(norm1dsignedl2(dtmp))        
    cf = np.concatenate(tfea,axis=1) 
    return cf
        
def loadmkt(lvid,pmkt):
    cf = readonemkt(pmkt,lvid[0])
    count = 0
    nvid = len(lvid)
    vfea = np.zeros((nvid,cf.shape[-1]))
    for i in range(nvid):    
        vfea[i,:] = readonemkt(pmkt,lvid[i])
        count += 1
        if (count+1) %100 == 0:
            print('[%d/%d]'%(count+1,nvid))
        sys.stdout.flush()
    return np.array(vfea)


def floadtsn(lvid,lpath):
    prgb=lpath[0]
    pflow=lpath[1]
    vfea = []
    vids = []
    count = 0
    nvid = len(lvid)
    for i in range(nvid):
        vid = lvid[i]
        vrgb = np.load('%s/%s.npy'%(prgb,vid))
        vflow = np.load('%s/%s.npy'%(pflow,vid))
        cf = np.concatenate([norm1dsignedl2(np.mean(vrgb,axis=0)),
                             norm1dsignedl2(np.std(vrgb,axis=0)),
                             norm1dsignedl2(np.mean(vflow,axis=0)),
                             norm1dsignedl2(np.std(vflow,axis=0))],axis=0)        
        vfea.append(cf)
        vids.append(vid)
        count += 1
        if (count+1) %100 == 0:
            print('[%d/%d]'%(count+1,nvid))
    return np.array(vfea),vids


def floademo(lvid,pemo):
    vfea = []
    vids = []
    count = 0
    nvid = len(lvid)
    for i in range(nvid):
        vid = lvid[i]
        vemo = readcsv('%s/%s.csv'%(pemo,vid))  
        cf = norm1dsignedl2(vemo)       
        vfea.append(cf)
        vids.append(vid)
        count += 1
        if (count+1) %100 == 0:
            print('[%d/%d]'%(count+1,nvid))
    return np.array(vfea),vids


def floadmkt(lvid,pmkt):
    vfea = []
    vids = []
    count = 0
    nvid = len(lvid)
    for i in range(nvid):
        vid = lvid[i]
        cf = np.squeeze(readonemkt(pmkt,lvid[i]),axis=0)    
        vfea.append(cf)
        vids.append(vid)
        count += 1
        if (count+1) %100 == 0:
            print('[%d/%d]'%(count+1,nvid))
    return np.array(vfea),vids

def readonehsh(pmkt,vid):
    ftmp = '%s/%s.gz'%(pmkt,vid)
    fs = cv2.FileStorage(ftmp,0)
    dtmp = fs.getNode('vocabulary').mat()
    if dtmp is None:
        print(vid)
    cf = norm1dsignedl2(dtmp)
    return cf
def floadhsh(lvid,phsh):
    vfea = []
    vids = []
    count = 0
    nvid = len(lvid)
    for i in range(nvid):
        vid = lvid[i]
        cf = np.squeeze(readonehsh(phsh,lvid[i]),axis=0)    
        vfea.append(cf)
        vids.append(vid)
        count += 1
        if (count+1) %1000 == 0:
            print('[%d/%d]'%(count+1,nvid))
    return np.array(vfea),vids

def floadmfcc(lvid,path):
    vfea = []
    vids = []
    count = 0
    nvid = len(lvid)
    for i in range(nvid):
        vid = lvid[i]
        fp = '%s/%s.mat'%(path,vid)
        cf = np.squeeze(sio.loadmat(fp)['enc_mfcc'])
        cf = norm1dsignedl2(cf)
        vfea.append(cf)
        vids.append(vid)
        count += 1
        if (count+1) %100 == 0:
            print('[%d/%d]'%(count+1,nvid))
    return np.array(vfea),vids

def loadfeas(lvid,lfeas,lpath,pres,ftype,dataset):
    lfv = []
    for f in lfeas:
        fpath = '%s/npy/%s_%s_%s.npy'%(pres,dataset,f,ftype)
        if os.path.exists(fpath):
            print('load %s...'%fpath)
            sys.stdout.flush()            
            vtest = np.load(fpath)
        else:
            if 'tsn' == f:
                vtest,vidtest = floadtsn(lvid,lpath[f])
            elif 'emo' == f or 'emolarge' == f or 'is13' == f:
                vtest,vidtest = floademo(lvid,lpath[f])
            elif 'mkt' == f:
                vtest,vidtest = floadmkt(lvid,lpath[f])
            elif 'hsh' == f:
                vtest,vidtest = floadhsh(lvid,lpath[f])
            elif 'mfcc' == f:
                vtest,vidtest = floadmfcc(lvid,lpath[f])
            elif 'dsift' == f:
                vtest,vidtest = floadhsh(lvid,lpath[f])
            else:
                print('[ERROR] unknow feature %s'%f)
                exit(0)
            if False == os.path.exists(fpath):
                np.save(fpath, vtest)
        lfv.append(vtest)
        
        print(f,vtest.shape)
    vfea = np.concatenate(lfv,axis=1)
    return vfea, None, None
