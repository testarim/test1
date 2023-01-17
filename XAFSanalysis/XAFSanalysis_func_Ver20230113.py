#Anaconda 4.12.0
#python 3.9.12
#xraylarch 0.9.59
#ipywidgets 7.6.5
import csv
import os
from pathlib import Path # added by fujikawa on 20222/12/23 
    
# 処理速度向上のため並列・平行処理を導入　 Implemented by fujikawa since 2022.Sep
import concurrent.futures # for threading on 2022.22.15
from functools import partial # for threading 

import time # added fot benchmark on 2202.11.15
import datetime # added fot benchmark on 2202.11.15
import cProfile, pstats# added fot benchmark on 2202.11.15
import io as iostream # added fot benchmark on 2202.11.15
from pstats import SortKey # added fot benchmark on 2202.11.15

import matplotlib.pyplot as plt
from ipywidgets import (interact, BoundedFloatText, interactive_output, HBox, VBox, Label,
                        Text, IntSlider, Layout, Checkbox, SelectMultiple, BoundedIntText,
                        Dropdown, Button, Textarea, Select, IntText)

# logファイルの作成  Implemented by fujikawa since 2022.Sep
# loggerの設定basicConfigはimport larch以上で行わないと動作しない
import logging
log_date = datetime.datetime.now()
logging.basicConfig(filename = f'XAFSanalysis_func_{log_date.strftime("%Y%m%d%H%M")}.log', encoding='utf-8', level=logging.DEBUG)

from larch import io, xafs, Group, Interpreter, xray
from larch.fitting import param, param_group
import larch as la
import glob
import itertools
import numpy as ny
import copy
from numba import jit # added for performance acceleration on 2022/12/27

logging.info(f'logger start on {log_date.strftime("%Y%m%d%H%M%S")}')

PREFIXES = ['norm', 'chik', 'FT', 'Fit']

# 評価機能のON（True）、OFF
PROFILER = False # for profiler on 2202.11.15
BENCH = True # for benchmark on 2202.11.15
#BENCH = False
#メモリー監視のため
MEMORY_ASSESSMENT = True # for momory monitor on 2022.11.17
#MEMORY_ASSESSMENT = False

# 最適化処理のON(True）、OFF(False)
#THREAD_SAVE = True # for threading of save text file on 2202.11.15
THREAD_SAVE = False 

BINARY_SAVE = True  # for save files as binary files on 2202.11.17
#BINARY_SAVE = False

OPTIMIZE_FIT = True # for initial value optimization of FEFFit variables  
#OPTIMIZE_FIT = False 

#THREAD_FIT = True # for threading of FEFFit()
THREAD_FIT = False 

THREAD_FT = True  # for threading of XAFSana_all()
#THREAD_FT = False  

# ファイルのテキストでのセーブとバイナリでのセーブは併用できない
if THREAD_SAVE and BINARY_SAVE is True:
    THREAD_SAVE = False 
    logging.warning("Only THREAD_SAVE was changed to False, because both THREAD_SAVE and BINARY_SAVE are True.\nYou may assign False to either of these.")
    
THREAD_CPU_CORE = 2 # a number of cpu cores provided with threading
THREAD_FT_CPU_CORE = 2 # a number of cpu cores provided with threading of XAFSana_all()


session=Interpreter()

#各種パラメーターの初期値
ipreE1=-200#pre-edge line range
ipreE2=-50
ipostE1=150#post-edge line range
ipostE2=400
inormO=3#normalization order
irbkg1=1.0#Rbkg
ikmin1=0.0#spline range in k
ikmax1=15.0
ikmin2=3.0#FT range in k
ikmax2=9.0
irmin1=1.0#Fitting range in r
irmax1=3.0
ikw=2#Fitting k-weight

F_initial=False#Paramsetとpathlistの初期値をargファイルの値にするかどうかの判定



#jupyternotebookで連続XAFSスペクトル解析するための関数

#pythonのホームディレクトリに以下のファイルを生成する。
#arg1.csv ファイルパス等
#arg2.csv　解析パラメータ
#arg3.csv　フィッティングパスの設定

#変数を渡すためのcsvファイル作成(もう使っていない)
def save_arg(a):
    with open('arg.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(a)
    return a

#ファイルパスとか保存したcsvファイルを読み込む
def read_arg1(): # modified by fujikawa on 2022/12/23 
    with open('arg1.csv','r') as f:
        file1=f.read()
    file = file1.split(',')
    filelist=glob.glob(file[0]+'/'+file[1]+file[2])
    filenamelist=[]
    savefileD=file[3].rstrip('\n')
    for fn in filelist:
        filenamelist.append(os.path.basename(fn))
    return filelist,filenamelist,savefileD

#解析パラメータ読み込む
def read_arg2():
    with open('arg2.csv','r') as f:
        file2=f.read()
    file=file2.split(',')
    readfile=file[0].split(':')
    preE1 = float(file[1])#pre edge range
    preE2 = float(file[2])
    postE1 = float(file[3])#normalization reange
    postE2 = float(file[4])
    e01 = float(file[5])#E0
    normO = float(file[6])#normalization order
    rbkg1 = float(file[7])
    kmin1 = float(file[8])#spline range in k
    kmax1 = float(file[9])
    kmin2 = float(file[10])#forward FT range in k
    kmax2 = float(file[11])
    rmin1 = float(file[12])#fitting range in r
    rmax1 = float(file[13])
    kw2 = float(file[14])#fitting k-weight
    return readfile,preE1,preE2,postE1,postE2,e01,normO,rbkg1,kmin1,kmax1,kmin2,kmax2,rmin1,rmax1,kw2

#フィッティングパスの設定読み込む
def read_arg3():
    with open('arg3.csv','r') as f:
        file=f.read().splitlines()
    No_path=int(file[0])#pathの数
    para_n=file[1].split(',')#変数の名前
    para_v=[float(n) for n in file[2].split(',')]#初期値
    para_min=[float(n) for n in file[3].split(',')]#最小値
    para_max=[float(n) for n in file[4].split(',')]#最大値
    para_va=[n=='True' for n in file[5].split(',')]#vary bool()だとだめだったので文字列の比較でブール値に変換する
    feff_l=file[6].split(',')#feffファイルのパス
    para2_u_list=[n=='True' for n in file[7].split(',')]#C3,C4,Eiを使用するか
    para2_n=file[8].split(',')#C3,C4,Eiの名前
    para2_v=[float(n) for n in file[9].split(',')]#C3,C4,Ei初期値
    para2_min=[float(n) for n in file[10].split(',')]#C3,C4,Ei最小値
    para2_max=[float(n) for n in file[11].split(',')]#C3,C4,Ei最大値
    para2_va=[n=='True' for n in file[12].split(',')]#C3,C4,Ei vary
    para_name=[para_n[0:5],para_n[5:10],para_n[10:15],para_n[15:20]]#2d listにする
    para_value=[para_v[0:5],para_v[5:10],para_v[10:15],para_v[15:20]]
    para_min=[para_min[0:5],para_min[5:10],para_min[10:15],para_min[15:20]]
    para_max=[para_max[0:5],para_max[5:10],para_max[10:15],para_max[15:20]]
    para_variable=[para_va[0:5],para_va[5:10],para_va[10:15],para_va[15:20]]
    para2_name=[para2_n[0:5],para2_n[5:10],para2_n[10:15]]
    para2_value=[para2_v[0:5],para2_v[5:10],para2_v[10:15]]
    para2_min=[para2_min[0:5],para2_min[5:10],para2_min[10:15]]
    para2_max=[para2_max[0:5],para2_max[5:10],para2_max[10:15]]
    para2_variable=[para2_va[0:5],para2_va[5:10],para2_va[10:15]]
    return No_path, para_name, para_value, para_min, para_max, para_variable, feff_l, para2_u_list, para2_name, para2_value, para2_min, para2_max, para2_variable

#フィッティングに使った変数を保存する
def save_args():
    with open('arg1.csv','r') as f:
        file1=f.read()
    file = file1.split(',')
    fileD=file[0]
    fileP=file[1]
    fileE=file[2]
    saveD=file[3].rstrip('\n')
    readfile,preE1,preE2,postE1,postE2,e01,normO,rbkg1,kmin1,kmax1,kmin2,kmax2,rmin1,rmax1,kw2=read_arg2()
    No_path, para_name, para_value, para_min, para_max, para_variable,feff_l,para2_u_l,para2_name,para2_value,para2_min,para2_max,para2_variable=read_arg3()
    para_name=para_name[0][0:No_path]+para_name[1][0:No_path]+para_name[2][0:No_path]+para_name[3][0:No_path]
    para_value=para_value[0][0:No_path]+para_value[1][0:No_path]+para_value[2][0:No_path]+para_value[3][0:No_path]
    para_min=para_min[0][0:No_path]+para_min[1][0:No_path]+para_min[2][0:No_path]+para_min[3][0:No_path]
    para_max=para_max[0][0:No_path]+para_max[1][0:No_path]+para_max[2][0:No_path]+para_max[3][0:No_path]
    para_variable=para_variable[0][0:No_path]+para_variable[1][0:No_path]+para_variable[2][0:No_path]+para_variable[3][0:No_path]
    feff_l=feff_l[0:No_path]
    if para2_u_l[0]:
        para_name.extend(para2_name[0][0:No_path])
        para_value.extend(para2_value[0][0:No_path])
        para_min.extend(para2_min[0][0:No_path])
        para_max.extend(para2_max[0][0:No_path])
        para_variable.extend(para2_variable[0][0:No_path])
    if para2_u_l[1]:
        para_name.extend(para2_name[1][0:No_path])
        para_value.extend(para2_value[1][0:No_path])
        para_min.extend(para2_min[1][0:No_path])
        para_max.extend(para2_max[1][0:No_path])
        para_variable.extend(para2_variable[1][0:No_path])
    if para2_u_l[2]:
        para_name.extend(para2_name[2][0:No_path])
        para_value.extend(para2_value[2][0:No_path])
        para_min.extend(para2_min[2][0:No_path])
        para_max.extend(para2_max[2][0:No_path])
        para_variable.extend(para2_variable[2][0:No_path])
    with open(saveD+'/parameters.csv','w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['E0','Normalization order','Pre-edge range','','Normalization range','','Rbkg','Spline range in k','','Fourier transform range','','Fitting range','','k-weight'])
        writer.writerow(['e01','normO','preE1','preE2','postE1','postE2','rbkg1','kmin1','kmax1','kmin2','kmax2','rmin1','rmax1','kw2'])
        writer.writerow([e01,normO,preE1,preE2,postE1,postE2,rbkg1,kmin1,kmax1,kmin2,kmax2,rmin1,rmax1,kw2])
        writer.writerow(['No of Paths'])
        writer.writerow([No_path])
        writer.writerow(['initial values'])
        writer.writerow(para_name)
        writer.writerow(para_value)
        writer.writerow(para_min)
        writer.writerow(para_max)
        writer.writerow(para_variable)
        writer.writerow(feff_l)
        writer.writerow(['file directory','filename pattern','filename extension', 'save directory'])
        writer.writerow([fileD, fileP, fileE, saveD])
    return rmin1

#encoderの値がおかしい場合線形補完する(WSとencoderの差の四分位範囲(の4倍)を基準に異常値を探す)
def interp_deg(aWS,aENC):
    c_aWS=copy.copy(aWS)#encoderの値がおかしい部分を削除した配列を作るためにコピーする
    c_aENC=copy.copy(aENC)
    dif_deg=[i-j for i, j in zip(c_aWS, c_aENC)]
    q1,q3=ny.percentile(dif_deg, [25,75])
    iqr=q3-q1
    lb=q1-(iqr*4)
    ub=q3+(iqr*4)
    x=ny.array(dif_deg)[((dif_deg < lb)| (dif_deg > ub))]
    i_del=[]
    for i,j in enumerate(x):
        i_del.append(int(ny.where(dif_deg == j)[0]))
    if len(i_del) != 0:
        i_del=i_del[::-1]
        for i, j in enumerate(i_del):
            del c_aWS[j]
            del c_aENC[j]
        c_aWS=c_aWS[::-1]#ny.interpは単調増加である必要があるのでひっくり返す
        c_aENC=c_aENC[::-1]
        for i, j in enumerate(i_del):
            aENC[j]=ny.interp(aWS[j],c_aWS,c_aENC)
    return aENC

#conventionalとDXAFSのデータを読み込めるようにする
#1行目に”  9809"(スペース2つ＋9809)があるとPFの9809フォーマットとして読み込む
#サンプル名に9809が入っていた場合.nemファイルでも9809フォーマットとなってしまうためスペース２つも条件文に入れた
def read_xafsdata(filepath, element=[-1]):
    with open(filepath) as f:
        file1line = f.readline()
    if file1line.find('  9809') != -1:
        with open(filepath) as f:
            reader = f.read()
        file = reader.splitlines()
        points = int(file[5].split('Points=')[1])
        block = int(file[6].split('Block =')[1])#ブロック数
        d = float(file[4][26:36])#面間隔
        offset_s = file[12+block].split('Offset')[1]
        offset = [float(offset_s[i*10:(i+1)*10]) for i in range(int(len(offset_s)/10))]#floatのlistに変換
        xmu,deg1,deg2 = [],[],[]
        out = Group()
        if file[5].find('Transmission') != -1:#transmissionのデータ読み込み
            for i in range(13+block, 13+block+points):
                deg2.append(float(file[i][10:20]))
                deg1.append(float(file[i][0:10]))
                xmu.append(ny.log((float(file[i][30:40])-offset[2])/(float(file[i][40:50])-offset[3])))
        elif file[5].find('Fluorescence') != -1:#Fluolessenceのデータ読み込み
            fluo=[]
            for i in range(13+block, 13+block+points):
                deg2.append(float(file[i][10:20]))
                deg1.append(float(file[i][0:10]))
                fluo=[float(file[i][j*10+30:j*10+40]) for j in range(36)]
                if element[0] != -1:#使用しない素子のデータを削除
                    element=element[::-1]
                    for k, kk in enumerate(element):
                        del fluo[kk]
                xmu.append(sum(fluo)/(float(file[i][390:400])-offset[38]))
        angle_encoder=interp_deg(deg1,deg2)
        out.e = ny.array([(12398.52/(2*d*ny.sin(angle/180*ny.pi))) for i,angle in enumerate(angle_encoder)])#エネルギーの配列
        out.xmu = ny.array(xmu)#吸収係数の配列
        out.array_labels = ['e','xmu']#以下はなくてもいいけど追加するattrsはよくわからないので無視する
        out.data = ny.stack([out.e,out.xmu])
        out.path = filepath
        out.header = '#' + file[2]
        out.filename = os.path.basename(filepath)
    else:#col1 energy(eV), col2 xmu .nemやDXAFSのデータ
        out=io.read_ascii(filepath, labels=['e','xmu'])
    return out

#arg1.csvとarg2.csvのパラメーターの通りにFTまで実行する。
def XAFSana(pkw):
    filelist,filenamelist,savefileD=read_arg1()
    readfile,preE1,preE2,postE1,postE2,e01,normO,rbkg1,kmin1,kmax1,kmin2,kmax2,rmin1,rmax1,kw2=read_arg2()
    if pkw == -1:#pkw=-1のときkw2で重み付けする
        pkw=kw2
    preEkws=dict(nnorm=normO,nvict=0,pre1=preE1,pre2=preE2,norm1=postE1,norm2=postE2)
    d=[]
    for rf in readfile:
        g=io.read_ascii(filelist[int(rf)],labels=['e','xmu'])
        xafs.autobk(g.e,g.xmu,rbkg=rbkg1,e0=e01,pre_edge_kws=preEkws,kmin=kmin1,kmax=kmax1,group=g)
        xafs.xftf(g,kmin=kmin2,kmax=kmax2,window='hanning',dk=1,kweight=pkw,group=g)
        d.append(g)
    return d

def calc_xftf(data_number, file, rbkg1, e01, preEkws, kmin1, kmax1, kmin2, kmax2, kw2):
    g=io.read_ascii(file,labels=['e','xmu'])
    xafs.autobk(g.e,g.xmu,rbkg=rbkg1,e0=e01,pre_edge_kws=preEkws,kmin=kmin1,kmax=kmax1,group=g)
    xafs.xftf(g,kmin=kmin2,kmax=kmax2,window='hanning',dk=1,kweight=kw2,group=g)
    return g

#arg1.csvで指定したすべてのファイルに対し、FTまで実行する
def XAFSana_all(ft_cpu_core ):
    if BENCH == True:
        with open("bench.txt",mode="a+") as bench_file: # for benchmark on 2202.10.25
            bench_file.write(f"XAFSana_all on {datetime.datetime.now()} \n") # for benchmark on 2202.10.25
        start_time_xafsana = time.perf_counter() # for benchmark on 2202.10.25
    filelist,filenamelist,savefileD=read_arg1()
    readfile,preE1,preE2,postE1,postE2,e01,normO,rbkg1,kmin1,kmax1,kmin2,kmax2,rmin1,rmax1,kw2=read_arg2()
    preEkws=dict(nnorm=normO,nvict=0,pre1=preE1,pre2=preE2,norm1=postE1,norm2=postE2)
    if THREAD_FT == True: # for threading of FT on 2022.11.25 
        logging.info("THREAD_FT start") 
        d=[0 for i in range(len(filelist))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=ft_cpu_core) as executor:
            futures = {executor.submit(calc_xftf, data_number, file, rbkg1, e01, preEkws, kmin1, kmax1, kmin2, kmax2, kw2 ):\
                                                 (data_number, file) for data_number, file in enumerate(filelist)}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                data_number, file_name = futures[future] # 最初に振り分けたthreadの番号と、filelistのイテレータを回収する 
                g = future.result()
                d[data_number] = g
    else:
        d=[]
        for file in filelist:
            g=io.read_ascii(file,labels=['e','xmu'])
            xafs.autobk(g.e,g.xmu,rbkg=rbkg1,e0=e01,pre_edge_kws=preEkws,kmin=kmin1,kmax=kmax1,group=g)
            xafs.xftf(g,kmin=kmin2,kmax=kmax2,window='hanning',dk=1,kweight=kw2,group=g)
            d.append(g)
    if BENCH == True:
        end_time_xafsana = time.perf_counter() # for benchmark on 2202.10.25
        differential_time_xafsana = end_time_xafsana - start_time_xafsana # for benchmark on 2202.10.25
        with open("bench.txt",mode="a+") as bench_file: # for benchmark on 2202.10.25
            bench_file.write(f"XAFS_ana_all are called in {differential_time_xafsana}  on {datetime.datetime.now()} \n") # for benchmark on 2202.10.25
    return d

#XAFSanaのデータのフィッティング
# modified by Fujikawa on 2022.Nov
def FEFFfit(d, checkbox_thread_fit_value, checkbox_optimize_fit_value, fit_cpu_core):
    readfile,preE1,preE2,postE1,postE2,e01,normO,rbkg1,kmin1,kmax1,kmin2,kmax2,rmin1,rmax1,kw2=read_arg2()
    No_path, para_name, para_value, para_min, para_max, para_variable, feff_l,para2_u_l,para2_name,para2_value,para2_min,para2_max,para2_variable=read_arg3()
    pars=param_group()
    n=0
    para2_u=para2_u_l[0]*1+para2_u_l[1]*2+para2_u_l[2]*4
    feff_path_list=[]
    if n < No_path:#1つめのpath
        a1=param(para_value[0][n],min=para_min[0][n],max=para_max[0][n],vary=para_variable[0][n])
        e1=param(para_value[1][n],min=para_min[1][n],max=para_max[1][n],vary=para_variable[1][n])
        r1=param(para_value[2][n],min=para_min[2][n],max=para_max[2][n],vary=para_variable[2][n])
        s1=param(para_value[3][n],min=para_min[3][n],max=para_max[3][n],vary=para_variable[3][n])
        pars.a1=a1
        pars.e1=e1
        pars.r1=r1
        pars.s1=s1
        if para2_u_l[0]:
            c31=param(para2_value[0][n],min=para2_min[0][n],max=para2_max[0][n],vary=para2_variable[0][n])
            pars.c31=c31
        if para2_u_l[1]:
            c41=param(para2_value[1][n],min=para2_min[1][n],max=para2_max[1][n],vary=para2_variable[1][n])
            pars.c41=c41
        if para2_u_l[2]:
            ei1=param(para2_value[2][n],min=para2_min[2][n],max=para2_max[2][n],vary=para2_variable[2][n])
            pars.ei1=ei1
        if para2_u == 0:
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a1',e0='e1',deltar='r1',sigma2='s1',_larch=session))
        elif para2_u == 1:#C3
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a1',e0='e1',deltar='r1',sigma2='s1',third='c31',_larch=session))
        elif para2_u == 2:#C4
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a1',e0='e1',deltar='r1',sigma2='s1',fourth='c41',_larch=session))
        elif para2_u == 3:#C3,C4
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a1',e0='e1',deltar='r1',sigma2='s1',third='c31',fourth='c41',_larch=session))
        elif para2_u == 4:#Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a1',e0='e1',deltar='r1',sigma2='s1',ei='ei1',_larch=session))
        elif para2_u == 5:#C3,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a1',e0='e1',deltar='r1',sigma2='s1',third='c31',ei='ei1',_larch=session))
        elif para2_u == 6:#C4,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a1',e0='e1',deltar='r1',sigma2='s1',fourth='c41',ei='ei1',_larch=session))
        elif para2_u == 7:#C3,C4,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a1',e0='e1',deltar='r1',sigma2='s1',third='c31',fourth='c41',ei='ei1',_larch=session))
        n=n+1
    if n < No_path:#2つ目のpath
        a2=param(para_value[0][n],min=para_min[0][n],max=para_max[0][n],vary=para_variable[0][n])
        e2=param(para_value[1][n],min=para_min[1][n],max=para_max[1][n],vary=para_variable[1][n])
        r2=param(para_value[2][n],min=para_min[2][n],max=para_max[2][n],vary=para_variable[2][n])
        s2=param(para_value[3][n],min=para_min[3][n],max=para_max[3][n],vary=para_variable[3][n])
        pars.a2=a2
        pars.e2=e2
        pars.r2=r2
        pars.s2=s2
        if para2_u_l[0]:
            c32=param(para2_value[0][n],min=para2_min[0][n],max=para2_max[0][n],vary=para2_variable[0][n])
            pars.c32=c32
        if para2_u_l[1]:
            c42=param(para2_value[1][n],min=para2_min[1][n],max=para2_max[1][n],vary=para2_variable[1][n])
            pars.c42=c42
        if para2_u_l[2]:
            ei2=param(para2_value[2][n],min=para2_min[2][n],max=para2_max[2][n],vary=para2_variable[2][n])
            pars.ei2=ei2
        if para2_u == 0:
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a2',e0='e2',deltar='r2',sigma2='s2',_larch=session))
        elif para2_u == 1:#C3
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a2',e0='e2',deltar='r2',sigma2='s2',third='c32',_larch=session))
        elif para2_u == 2:#C4
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a2',e0='e2',deltar='r2',sigma2='s2',fourth='c42',_larch=session))
        elif para2_u == 3:#C3,C4
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a2',e0='e2',deltar='r2',sigma2='s2',third='c32',fourth='c42',_larch=session))
        elif para2_u == 4:#Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a2',e0='e2',deltar='r2',sigma2='s2',ei='ei2',_larch=session))
        elif para2_u == 5:#C3,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a2',e0='e2',deltar='r2',sigma2='s2',third='c32',ei='ei2',_larch=session))
        elif para2_u == 6:#C4,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a2',e0='e2',deltar='r2',sigma2='s2',fourth='c42',ei='ei2',_larch=session))
        elif para2_u == 7:#C3,C4,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a2',e0='e2',deltar='r2',sigma2='s2',third='c32',fourth='c42',ei='ei2',_larch=session))
        n=n+1
    if n < No_path:#3つ目のpath
        a3=param(para_value[0][n],min=para_min[0][n],max=para_max[0][n],vary=para_variable[0][n])
        e3=param(para_value[1][n],min=para_min[1][n],max=para_max[1][n],vary=para_variable[1][n])
        r3=param(para_value[2][n],min=para_min[2][n],max=para_max[2][n],vary=para_variable[2][n])
        s3=param(para_value[3][n],min=para_min[3][n],max=para_max[3][n],vary=para_variable[3][n])
        pars.a3=a3
        pars.e3=e3
        pars.r3=r3
        pars.s3=s3
        if para2_u_l[0]:
            c33=param(para2_value[0][n],min=para2_min[0][n],max=para2_max[0][n],vary=para2_variable[0][n])
            pars.c33=c33
        if para2_u_l[1]:
            c43=param(para2_value[1][n],min=para2_min[1][n],max=para2_max[1][n],vary=para2_variable[1][n])
            pars.c43=c43
        if para2_u_l[2]:
            ei3=param(para2_value[2][n],min=para2_min[2][n],max=para2_max[2][n],vary=para2_variable[2][n])
            pars.ei3=ei3
        if para2_u == 0:
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a3',e0='e3',deltar='r3',sigma2='s3',_larch=session))
        elif para2_u == 1:#C3
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a3',e0='e3',deltar='r3',sigma2='s3',third='c33',_larch=session))
        elif para2_u == 2:#C4
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a3',e0='e3',deltar='r3',sigma2='s3',fourth='c43',_larch=session))
        elif para2_u == 3:#C3,C4
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a3',e0='e3',deltar='r3',sigma2='s3',third='c33',fourth='c43',_larch=session))
        elif para2_u == 4:#Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a3',e0='e3',deltar='r3',sigma2='s3',ei='ei3',_larch=session))
        elif para2_u == 5:#C3,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a3',e0='e3',deltar='r3',sigma2='s3',third='c33',ei='ei3',_larch=session))
        elif para2_u == 6:#C4,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a3',e0='e3',deltar='r3',sigma2='s3',fourth='c43',ei='ei3',_larch=session))
        elif para2_u == 7:#C3,C4,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a3',e0='e3',deltar='r3',sigma2='s3',third='c33',fourth='c43',ei='ei3',_larch=session))
        n=n+1
    if n < No_path:#4つ目のpath
        a4=param(para_value[0][n],min=para_min[0][n],max=para_max[0][n],vary=para_variable[0][n])
        e4=param(para_value[1][n],min=para_min[1][n],max=para_max[1][n],vary=para_variable[1][n])
        r4=param(para_value[2][n],min=para_min[2][n],max=para_max[2][n],vary=para_variable[2][n])
        s4=param(para_value[3][n],min=para_min[3][n],max=para_max[3][n],vary=para_variable[3][n])
        pars.a4=a4
        pars.e4=e4
        pars.r4=r4
        pars.s4=s4
        if para2_u_l[0]:
            c34=param(para2_value[0][n],min=para2_min[0][n],max=para2_max[0][n],vary=para2_variable[0][n])
            pars.c34=c34
        if para2_u_l[1]:
            c44=param(para2_value[1][n],min=para2_min[1][n],max=para2_max[1][n],vary=para2_variable[1][n])
            pars.c44=c44
        if para2_u_l[2]:
            ei4=param(para2_value[2][n],min=para2_min[2][n],max=para2_max[2][n],vary=para2_variable[2][n])
            pars.ei4=ei4
        if para2_u == 0:
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a4',e0='e4',deltar='r4',sigma2='s4',_larch=session))
        elif para2_u == 1:#C3
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a4',e0='e4',deltar='r4',sigma2='s4',third='c34',_larch=session))
        elif para2_u == 2:#C4
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a4',e0='e4',deltar='r4',sigma2='s4',fourth='c44',_larch=session))
        elif para2_u == 3:#C3,C4
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a4',e0='e4',deltar='r4',sigma2='s4',third='c34',fourth='c44',_larch=session))
        elif para2_u == 4:#Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a4',e0='e4',deltar='r4',sigma2='s4',ei='ei4',_larch=session))
        elif para2_u == 5:#C3,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a4',e0='e4',deltar='r4',sigma2='s4',third='c34',ei='ei4',_larch=session))
        elif para2_u == 6:#C4,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a4',e0='e4',deltar='r4',sigma2='s4',fourth='c44',ei='ei4',_larch=session))
        elif para2_u == 7:#C3,C4,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a4',e0='e4',deltar='r4',sigma2='s4',third='c34',fourth='c44',ei='ei4',_larch=session))
        n=n+1
    if n < No_path:#5つ目のpath
        a5=param(para_value[0][n],min=para_min[0][n],max=para_max[0][n],vary=para_variable[0][n])
        e5=param(para_value[1][n],min=para_min[1][n],max=para_max[1][n],vary=para_variable[1][n])
        r5=param(para_value[2][n],min=para_min[2][n],max=para_max[2][n],vary=para_variable[2][n])
        s5=param(para_value[3][n],min=para_min[3][n],max=para_max[3][n],vary=para_variable[3][n])
        pars.a5=a5
        pars.e5=e5
        pars.r5=r5
        pars.s5=s5
        if para2_u_l[0]:
            c35=param(para2_value[0][n],min=para2_min[0][n],max=para2_max[0][n],vary=para2_variable[0][n])
            pars.c35=c35
        if para2_u_l[1]:
            c45=param(para2_value[1][n],min=para2_min[1][n],max=para2_max[1][n],vary=para2_variable[1][n])
            pars.c45=c45
        if para2_u_l[2]:
            ei5=param(para2_value[2][n],min=para2_min[2][n],max=para2_max[2][n],vary=para2_variable[2][n])
            pars.ei5=ei5
        if para2_u == 0:
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a5',e0='e5',deltar='r5',sigma2='s5',_larch=session))
        elif para2_u == 1:#C3
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a5',e0='e5',deltar='r5',sigma2='s5',third='c35',_larch=session))
        elif para2_u == 2:#C4
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a5',e0='e5',deltar='r5',sigma2='s5',fourth='c45',_larch=session))
        elif para2_u == 3:#C3,C4
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a5',e0='e5',deltar='r5',sigma2='s5',third='c35',fourth='c45',_larch=session))
        elif para2_u == 4:#Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a5',e0='e5',deltar='r5',sigma2='s5',ei='ei5',_larch=session))
        elif para2_u == 5:#C3,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a5',e0='e5',deltar='r5',sigma2='s5',third='c35',ei='ei5',_larch=session))
        elif para2_u == 6:#C4,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a5',e0='e5',deltar='r5',sigma2='s5',fourth='c45',ei='ei5',_larch=session))
        elif para2_u == 7:#C3,C4,Ei
            feff_path_list.append(xafs.feffpath(feff_l[n],s02='a5',e0='e5',deltar='r5',sigma2='s5',third='c35',fourth='c45',ei='ei5',_larch=session))
    trans=xafs.feffit_transform(fitspace='r',kmin=kmin2,kmax=kmax2,dk=1,kweight=kw2,rmin=rmin1,rmax=rmax1,window='hanning')
    out_l=ny.zeros((No_path, len(d), 15))#outの中から必要なパラメータを出力する。欲しい値は同じidのオブジェクトなので書き換えられてしまうため。
    
#    if THREAD_FIT and OPTIMIZE_FIT  is True:
    if checkbox_thread_fit_value and checkbox_optimize_fit_value is True:
        logging.info('thread and optimize fit was ON')
        g2=[0 for _ in range(len(d))]
        dataset_l=[0 for _ in range(len(d))]
        # 最初のデータのみフィット変数の最適化を行う
        flag_opt = True
        _thread_calc_fefffit(d[0], feff_path_list, trans, session, pars, para_min, para_max, para_variable, n, No_path, 0, g2, out_l, dataset_l, flag_opt, checkbox_optimize_fit_value )    
        # 2番目以降のデータをスレッド処理
        flag_opt = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=fit_cpu_core) as executor:
            future_make_feffit_results = {executor.submit(_thread_calc_fefffit, sliced_d, feff_path_list, trans, session, pars, para_min, para_max,\
                                                        para_variable, n,  No_path, data_number+1, g2, out_l, dataset_l, flag_opt, checkbox_optimize_fit_value):(data_number, sliced_d)\
                                                        for data_number, sliced_d in enumerate(d[1:])}
            for future in concurrent.futures.as_completed(future_make_feffit_results):
                data_number, sliced_d = future_make_feffit_results[future] # 最初に振り分けたthreadの番号と、dのイテレータを回収する 
                future.result()
        
#    elif THREAD_FIT == True:
    elif checkbox_thread_fit_value is True:
        logging.info("THREAD_FIT start")
        print('thread fit was ON ')
        g2=[0 for _ in range(len(d))]
        dataset_l=[0 for _ in range(len(d))]
        logging.info(f" Start threadlog  on {datetime.datetime.now()} ") 
        with concurrent.futures.ThreadPoolExecutor(max_workers=fit_cpu_core) as executor:
            future_makeFeffitResults = {executor.submit(_thread_calc_fefffit, sliced_d, feff_path_list, trans, session, pars, para_min, para_max,\
                                                        para_variable, n,  No_path, data_number, g2, out_l, dataset_l, checkbox_optimize_fit_value):(data_number, sliced_d)\
                                                        for data_number, sliced_d in enumerate(d)}
            for i, future in enumerate(concurrent.futures.as_completed(future_makeFeffitResults)):
                data_number, sliced_d = future_makeFeffitResults[future] # 最初に振り分けたthreadの番号と、dのイテレータを回収する 
                future.result()
        
    else:
        logging.info('Normal fit was ON ')
        g2=[0 for _ in range(len(d))]
        dataset_l=[0 for _ in range(len(d))]
        flag_opt = True
        _thread_calc_fefffit(d[0], feff_path_list, trans, session, pars, para_min, para_max, para_variable, n, No_path, 0, g2, out_l, dataset_l, flag_opt, checkbox_optimize_fit_value)       
        flag_opt = False
        for i, g in enumerate(d[1:]): 
            _thread_calc_fefffit(g, feff_path_list, trans, session, pars, para_min, para_max, para_variable, n, No_path, i+1, g2, out_l, dataset_l, flag_opt, checkbox_optimize_fit_value)
        
    return g2, out_l, dataset_l

#@jit
def _optimize_fit_variables(fitout, pars, para_min, para_max, para_variable, n ): 
    """ 
        reassignment variable values of last fitting results to pars
    """
    pars.a1=param(fitout.params['a1'].value, min=para_min[0][n], max=para_max[0][n], vary=para_variable[0][n])
    pars.e1=param(fitout.params['e1'].value, min=para_min[1][n], max=para_max[1][n], vary=para_variable[1][n])
    pars.r1=param(fitout.params['r1'].value, min=para_min[2][n], max=para_max[2][n], vary=para_variable[2][n])
    pars.s1=param(fitout.params['s1'].value, min=para_min[3][n], max=para_max[3][n], vary=para_variable[3][n])
    
@jit
def _thread_calc_fefffit(d, feff_path_list, trans, session, pars, para_min, para_max, para_variable, n, No_path, data_number, g2, out_l, dataset_l, flag_opt, checkbox_optimize_fit_value):    
    dset=xafs.feffit_dataset(data=d,pathlist=feff_path_list,transform=trans,_larch=session)
    fitout=xafs.feffit(pars,dset,_larch=session)
#    if OPTIMIZE_FIT == True: 
    if checkbox_optimize_fit_value is True:
        if flag_opt is True:
            _optimize_fit_variables(fitout, pars, para_min, para_max, para_variable, n )
    g2[data_number]=d
    for np in range(No_path):#out_l[pathlist][datalist][paramlist]の3次元配列out_l[:,i,8]はR-factor(同じ値)
        ll=feff_path_list[np].label
        _assgin_results_out_l(np, data_number, out_l, fitout, ll)
        dataset_l[data_number]=dset
    # メモリ節約のため利用しないオブジェクトは消去
    #del dset, fitout, ll, par
    
#@jit
def _assgin_results_out_l(np, data_number, out_l, fitout, ll):
    par=fitout.params 
    out_l[np][data_number][0]=par.get('s02_'+ll).value
    out_l[np][data_number][1]=par.get('e0_'+ll).value
    out_l[np][data_number][2]=par.get('deltar_'+ll).value
    out_l[np][data_number][3]=par.get('sigma2_'+ll).value
    out_l[np][data_number][4]=par.get('s02_'+ll).stderr
    out_l[np][data_number][5]=par.get('e0_'+ll).stderr
    out_l[np][data_number][6]=par.get('deltar_'+ll).stderr
    out_l[np][data_number][7]=par.get('sigma2_'+ll).stderr
    out_l[np][data_number][8]=fitout.rfactor
    out_l[np][data_number][9]=par.get('third_'+ll).value
    out_l[np][data_number][10]=par.get('fourth_'+ll).value
    out_l[np][data_number][11]=par.get('ei_'+ll).value
    out_l[np][data_number][12]=par.get('third_'+ll).stderr
    out_l[np][data_number][13]=par.get('fourth_'+ll).stderr
    out_l[np][data_number][14]=par.get('ei_'+ll).stderr#xmu,chik,FT,fittingのグラフ描画class plot_bkgchift:
    

#xmu,chik,FT,fittingのグラフ描画    

class plot_bkgchift:
    def __init__(self):
        self.fig = plt.figure()
        def plotter(select):
            if select == 'xmu':#xmu選択時、色々表示選択チェックボックス作る
                def plotxmu(Norm,Pre_edge_line,Post_edge_line,Background,Emin,Emax):
                    self.fig.clear()
                    
                    d=XAFSana(2)
                    for g in d:
                        if Norm:#NormにチェックついているときはNormのみ描画
                            plt.plot(g.e,g.flat,color='b')
                        else:
                            if Pre_edge_line:
                                plt.plot(g.e,g.pre_edge,color='g')
                            if Post_edge_line:
                                plt.plot(g.e,g.post_edge,color='m')
                            if Background:
                                plt.plot(g.e,g.bkg,color='r')
                            plt.plot(g.e,g.xmu,color='b')
                    plt.xlim(g.e0+Emin,g.e0+Emax)#グラフ描画範囲
                C1=Checkbox(value=True, description='Pre edge line')
                C2=Checkbox(value=True, description='Post edge line')
                C3=Checkbox(value=True, description='Background')
                C4=Checkbox(value=False, description='Normalization')
                F1=BoundedFloatText(value=-200,min=-5000,max=5000,description='Emin')
                F2=BoundedFloatText(value=800,min=-5000,max=5000,description='Emax')
                out=interactive_output(plotxmu,{'Pre_edge_line':C1,'Post_edge_line':C2,'Background':C3,'Norm':C4,'Emin':F1,'Emax':F2})
                h1=HBox([F1,F2])#レイアウトをいじるのでinteractive_outputとdisplayを使う
                ui=VBox([C1,C2,C3,C4,h1])
                display(ui,out)
            elif select == 'chik':#スライダーで選択したkweighatのchik表示
                def plotchi(kw,KMIN,KMAX):
                    self.fig.clear()
                    d=XAFSana(2)
                    for g in d:
                        plt.plot(g.k,g.chi*(g.k**kw),color='b')
                    plt.xlim(KMIN,KMAX)
                S1=IntSlider(value=2,min=0,max=3,step=1,description='plotting k-weight')
                F1=BoundedFloatText(value=0,min=0,max=20,description='kmin')
                F2=BoundedFloatText(value=15,min=0,max=20,description='kmax')
                h1=HBox([F1,F2])
                ui=VBox([S1,h1])
                out=interactive_output(plotchi,{'kw':S1,'KMIN':F1,'KMAX':F2})
                display(ui,out)
            elif select == 'FT':
                def plotFT(Magnitude,Real_part,Imag_part,Rmin,Rmax,pkw):
                    self.fig.clear()
                    d=XAFSana(pkw)
                    for g in d:
                        if Magnitude:
                            plt.plot(g.r,g.chir_mag,color='b')
                        if Real_part:
                            plt.plot(g.r,g.chir_re,color='g')
                        if Imag_part:
                            plt.plot(g.r,g.chir_im,color='m')
                    plt.xlim(Rmin,Rmax)
                C1=Checkbox(value=True,description='Magnitude')
                C2=Checkbox(value=False,description='Real part')
                C3=Checkbox(value=False,description='Imag. part')
                F1=BoundedFloatText(value=0,min=0,max=20,description='Rmin')
                F2=BoundedFloatText(value=6,min=0,max=20,description='Rmax')
                S1=IntSlider(value=2,min=0,max=3,step=1,description='plotting k-weight')
                out=interactive_output(plotFT,{'Magnitude':C1,'Real_part':C2,'Imag_part':C3,'Rmin':F1,'Rmax':F2,'pkw':S1})
                h1=HBox([F1,F2])
                ui=VBox([S1,C1,C2,C3,h1])
                display(ui,out)
            elif select == 'Fitting':
                def plotfit(Rmin,Rmax):
                    self.fig.clear()
                    d=XAFSana(-1)
                    d2,d_out,d_dset=FEFFfit(d, checkbox_thread_fit.value, checkbox_optimize_fit.value, inttext_thread_core.value)
                    filenamelist=read_arg1()[1]
                    No_path=read_arg3()[0]
                    txt_out=''
                    for g, dset in itertools.product(d2,d_dset):
                        plt.plot(g.r,g.chir_mag)
                        plt.plot(dset.model.r,dset.model.chir_mag)
                    plt.xlim(Rmin,Rmax)
                    for i in range(len(d2)):
                        txt_out=txt_out + filenamelist[i] + ' R-factor=' + format(d_out[0,i,8]*100,'.2f') + '\n'
                        for j in range(No_path):
                            txt_out=(txt_out + 'Path' + str(j)
                            + ':S02=' + format(d_out[j,i,0],'.2f') + '+- ' + format(d_out[j,i,4], '.2f')
                            + ':E0=' + format(d_out[j,i,1],'.2f') + '+- ' + format(d_out[j,i,5], '.2f')
                            + ':deltar=' + format(d_out[j,i,2],'.2f') + '+- ' + format(d_out[j,i,6], '.2f')
                            + ':sigma2=' + format(d_out[j,i,3],'.2f') + '+- ' + format(d_out[j,i,7], '.2f')
                            + ':third=' + format(d_out[j,i,9],'.2f') + '+- ' + format(d_out[j,i,12], '.2f')
                            + ':fourth=' + format(d_out[j,i,10],'.2f') + '+- ' + format(d_out[j,i,13], '.2f')
                            + ':ei=' + format(d_out[j,i,11],'.2f') + '+- ' + format(d_out[j,i,14], '.2f')
                            + '\n'
                            )
                    T.value=txt_out
                F1=BoundedFloatText(value=0, min=0, max=10, description='Rmin')
                F2=BoundedFloatText(value=6, min=0, max=10, description='Rmax')
                T=Textarea(value='fitting results', description='fitting results', layout=Layout(width='500px',height='200px'))
                h1=HBox([F1,F2])
                ui=VBox([h1,T])
                out=interactive_output(plotfit,{'Rmin':F1,'Rmax':F2})
                display(ui,out)
        #xmu,chik,FTのdropdown生成
        interact(plotter, select=['xmu','chik','FT','Fitting'])#HBoxとかでレイアウトをイジる必要がなければinteractのみで済む#読み込むファイルと保存先の設定class dataDset:
#読み込むファイルと保存先の設定
class dataDset:
    def __init__(self,fileD,fileP,fileE,savefileD):
        def savefilepara(fileD,fileP,fileE,savefileD):
            with open('arg1.csv','w') as f:
                writer=csv.writer(f)
                writer.writerow([fileD,fileP,fileE,savefileD])
        T1 = Text(value = fileD, description='')
        T2 = Text(value = fileP, description = '', layout=Layout(width='100px'))
        T3 = Text(value = fileE, description = '', layout=Layout(width='100px'))
        T4 = Text(value = savefileD, description = '')
        L1=Label(value='data directory')
        L2=Label(value='data filename pattern')
        L3=Label(value='data filename extension')
        L4=Label(value='save directory')
        out = interactive_output(savefilepara, {'fileD':T1, 'fileP':T2, 'fileE':T3, 'savefileD':T4})
        h1=HBox([L1,T1,L2,T2,L3,T3])
        h2=HBox([L4,T4])
        ui=VBox([h1,h2])
        display(ui,out)
#パラメーターを決める
class Paramset:
    def __init__(self):
        filelist,filenamelist,savefileD=read_arg1()
        global F_initial
        ipreE1=-200#pre-edge line range
        ipreE2=-50
        ipostE1=150#post-edge line range
        ipostE2=400
        inormO=3#normalization order
        irbkg1=1.0#Rbkg
        ikmin1=0.0#spline range in k
        ikmax1=15.0
        ikmin2=3.0#FT range in k
        ikmax2=9.0
        irmin1=1.0#Fitting range in r
        irmax1=3.0
        ikw=2#Fitting k-weight
        g=io.read_ascii(filelist[0], labels=['e','xmu'])
        ee0=xafs.find_e0(g.e,g.xmu)
        if os.path.isfile('arg2.csv') == True:#arg2.csvが存在し、同じ吸収端であれば初期値とする。
            i2readfile,i2preE1,i2preE2,i2postE1,i2postE2,i2e01,i2normO,i2rbkg1,i2kmin1,i2kmax1,i2kmin2,i2kmax2,i2rmin1,i2rmax1,i2kw=read_arg2()
            if xray.guess_edge(ee0) == xray.guess_edge(i2e01):                                        
                ipreE1,ipreE2,ipostE1,ipostE2,inormO,irbkg1,ikmin1,ikmax1,ikmin2,ikmax2,irmin1,irmax1,\
                ikw=i2preE1,i2preE2,i2postE1,i2postE2,i2normO,i2rbkg1,i2kmin1,i2kmax1,i2kmin2,i2kmax2,i2rmin1,i2rmax1,i2kw
                F_initial=True
            else:
                F_initial=False
        else:
            F_initial=False
        def saveparam(preE1,preE2,postE1,postE2,e01,normO,rbkg1,kmin1,kmax1,kmin2,kmax2,readfilelist,rmin1,rmax1,kw2):
            rfl1=[]
            rf2=''
            for rf1 in readfilelist:
                rfl1.append(str(filenamelist.index(rf1)))
            rf2=':'.join(rfl1)
            with open('arg2.csv','w') as f:
                writer=csv.writer(f)
                writer.writerow([rf2,preE1,preE2,postE1,postE2,e01,normO,rbkg1,kmin1,kmax1,kmin2,kmax2,rmin1,rmax1,kw2])
        F2=BoundedFloatText(value=ipreE1, min=-1000, max=0, description='', layout=Layout(width='100px'))#preE1
        F3=BoundedFloatText(value=ipreE2, min=-1000, max=0, description='', layout=Layout(width='100px'))#preE2
        F4=BoundedFloatText(value=ipostE1, min=0, max=1000, description='', layout=Layout(width='100px'))#postE1
        F5=BoundedFloatText(value=ipostE2, min=0, max=1000, description='', layout=Layout(width='100px'))#postE2
        F6=BoundedFloatText(value=ee0,min=-1, max=150000, description='', layout=Layout(width='100px'))#e01
        F7=IntSlider(value=inormO,min=1, max=3,step=1,description='', layout=Layout(width='200px'))#normO
        F8=BoundedFloatText(value=irbkg1, min=0, max=5, description='', layout=Layout(width='50px'))#rbkg1
        F9=BoundedFloatText(value=ikmin1, min=0, max=20, description='', layout=Layout(width='100px'))#kmin1
        F10=BoundedFloatText(value=ikmax1,min=0, max=20, description='', layout=Layout(width='100px'))#kmax1
        F11=BoundedFloatText(value=ikmin2, min=0, max=20, description='', layout=Layout(width='100px'))#kmin2
        F12=BoundedFloatText(value=ikmax2, min=0, max=20, description='', layout=Layout(width='100px'))#kmax2
        F13=BoundedFloatText(value=irmin1, min=0, max=10, description='', layout=Layout(width='100px'))#rmin1
        F14=BoundedFloatText(value=irmax1, min=0, max=10, description='', layout=Layout(width='100px'))#rmax1
        F15=BoundedFloatText(value=ikw, min=0, max=4, description='', layout=Layout(width='50px'))#kw2
        S1=SelectMultiple(options=filenamelist,value=[filenamelist[0]], layout=Layout(height='200px'))
        L2=Label(value='Normalization order')
        L3=Label(value='Pre-edge range')
        L4=Label(value='Normalization range:')
        L5=Label(value='Spline range in k  :')
        L6=Label(value='FT k-range         :')
        L7=Label(value='to')
        L8=Label(value='E0')
        L9=Label(value='Rbkg')
        L10=Label(value='Fitting R-range')
        L11=Label(value='kw')
        out=interactive_output(saveparam,{'preE1':F2,'preE2':F3,'postE1':F4,'postE2':F5,'e01':F6,
                                            'normO':F7,'rbkg1':F8,'kmin1':F9,'kmax1':F10,'kmin2':F11,'kmax2':F12,
                                            'readfilelist':S1,'rmin1':F13,'rmax1':F14,'kw2':F15})
        h2=HBox([L8,F6,L2,F7])
        h3=HBox([L3,F2,L7,F3])
        h4=HBox([L4,F4,L7,F5])
        h5=HBox([L9,F8])
        h6=HBox([L5,F9,L7,F10])
        h7=HBox([L6,F11,L7,F12])
        h8=HBox([L10,F13,L7,F14,L11,F15])
        v1=VBox([h2,h3,h4,h5,h6,h7,h8])
        ui=HBox([v1,S1])
        display(ui,out)

#pathの設定
class pathlist:
    def __init__(self):
        global F_initial
        p_list=['path1','path2','path3','path4','path5']
        if os.path.isfile('arg3.csv') and F_initial:
            No_path, para_name, para_value, para_min, para_max, para_variable, feff_l,para2_u_l,para2_name,para2_value,para2_min,para2_max,para2_variable=read_arg3()
            a_n=para_name[0][:]
            e_n=para_name[1][:]
            r_n=para_name[2][:]
            s_n=para_name[3][:]
            c3_n=para2_name[0][:]
            c4_n=para2_name[1][:]
            ei_n=para2_name[2][:]
            a_v,e_v,r_v,s_v,c3_v,c4_v,ei_v=para_value[0][:],para_value[1][:],para_value[2][:],para_value[3][:],para2_value[0][:],para2_value[1][:],para2_value[2][:]
            a_va,e_va,r_va,s_va,c3_va,c4_va,ei_va=para_variable[0][:],para_variable[1][:],para_variable[2][:],para_variable[3][:],para2_variable[0][:],para2_variable[1][:],para2_variable[2][:]
            a_min,e_min,r_min,s_min,c3_min,c4_min,ei_min=para_min[0][:],para_min[1][:],para_min[2][:],para_min[3][:],para2_min[0][:],para2_min[1][:],para2_min[2][:]
            a_max,e_max,r_max,s_max,c3_max,c4_max,ei_max=para_max[0][:],para_max[1][:],para_max[2][:],para_max[3][:],para2_max[0][:],para2_max[1][:],para2_max[2][:]
        else:
            No_path=1
            a_n=['amp1','amp2','amp3','amp4','amp5']
            e_n=['e01','e02','e03','e04','e05']
            r_n=['delr1','delr2','delr3','delr4','delr5']
            s_n=['ss1','ss2','ss3','ss4','ss5']
            c3_n=['C3_1','C3_2','C3_3','C3_4','C3_5']
            c4_n=['C4_1','C4_2','C4_3','C4_4','C4_5']
            ei_n=['Ei_1','Ei_2','Ei_3','Ei_4','Ei_5']
            a_v,e_v,r_v,s_v,c3_v,c4_v,ei_v=[1]*5,[0]*5,[0]*5,[0.003]*5,[0]*5,[0]*5,[0]*5
            a_va,e_va,r_va,s_va,c3_va,c4_va,ei_va=[True]*5,[True]*5,[True]*5,[True]*5,[True]*5,[True]*5,[True]*5
            a_min,e_min,r_min,s_min,c3_min,c4_min,ei_min=[0]*5,[-10]*5,[-2]*5,[0]*5,[-10]*5,[-10]*5,[-10]*5
            a_max,e_max,r_max,s_max,c3_max,c4_max,ei_max=[20]*5,[10]*5,[2]*5,[1]*5,[10]*5,[10]*5,[10]*5
            feff_l=['feff0001.dat']*5
            para2_u_l=[False]*3
        F_initial = False
        def path_set(No_path,pathname,u_c3,u_c4,u_ei):
            def path_param_set(a_value,e_value,r_value,s_value,
                                a_minimum,e_minimum,r_minimum,s_minimum,
                                a_maximum,e_maximum,r_maximum,s_maximum,
                                a_variable,e_variable,r_variable,s_variable,
                                a_name,e_name,r_name,s_name,feffdat_path,
                                c3_value,c3_minimum,c3_maximum,c3_variable,
                                c4_value,c4_minimum,c4_maximum,c4_variable,
                                ei_value,ei_minimum,ei_maximum,ei_variable,
                                c3_name,c4_name,ei_name):
                a_n[p_index]=a_name
                e_n[p_index]=e_name
                r_n[p_index]=r_name
                s_n[p_index]=s_name
                a_v[p_index]=a_value
                e_v[p_index]=e_value
                r_v[p_index]=r_value
                s_v[p_index]=s_value
                a_min[p_index]=a_minimum
                e_min[p_index]=e_minimum
                r_min[p_index]=r_minimum
                s_min[p_index]=s_minimum
                a_max[p_index]=a_maximum
                e_max[p_index]=e_maximum
                r_max[p_index]=r_maximum
                s_max[p_index]=s_maximum
                a_va[p_index]=a_variable
                e_va[p_index]=e_variable
                r_va[p_index]=r_variable
                s_va[p_index]=s_variable
                feff_l[p_index]=feffdat_path
                u_list=[u_c3,u_c4,u_ei]
                if u_c3 == True:
                    c3_n[p_index]=c3_name
                    c3_v[p_index]=c3_value
                    c3_min[p_index]=c3_minimum
                    c3_max[p_index]=c3_maximum
                    c3_va[p_index]=c3_variable
                if u_c4 == True:
                    c4_n[p_index]=c4_name
                    c4_v[p_index]=c4_value
                    c4_min[p_index]=c4_minimum
                    c4_max[p_index]=c4_maximum
                    c4_va[p_index]=c4_variable
                if u_ei == True:
                    ei_n[p_index]=ei_name
                    ei_v[p_index]=ei_value
                    ei_min[p_index]=ei_minimum
                    ei_max[p_index]=ei_maximum
                    ei_va[p_index]=ei_variable
                with open('arg3.csv','w',newline='') as f:
                    writer=csv.writer(f)
                    writer.writerow([No_path])
                    writer.writerow(a_n+e_n+r_n+s_n)
                    writer.writerow(a_v+e_v+r_v+s_v)
                    writer.writerow(a_min+e_min+r_min+s_min)
                    writer.writerow(a_max+e_max+r_max+s_max)
                    writer.writerow(a_va+e_va+r_va+s_va)
                    writer.writerow(feff_l)
                    writer.writerow(u_list)
                    writer.writerow(c3_n+c4_n+ei_n)
                    writer.writerow(c3_v+c4_v+ei_v)
                    writer.writerow(c3_min+c4_min+ei_min)
                    writer.writerow(c3_max+c4_max+ei_max)
                    writer.writerow(c3_va+c4_va+ei_va)
            p_index=p_list.index(pathname)
            T1=Text(value=a_n[p_index])#名前
            T2=Text(value=e_n[p_index])
            T3=Text(value=r_n[p_index])
            T4=Text(value=s_n[p_index])
            F1=BoundedFloatText(value=a_v[p_index],min=-100,max=100)#初期値
            F2=BoundedFloatText(value=e_v[p_index],min=-100,max=100)
            F3=BoundedFloatText(value=r_v[p_index],min=-100,max=100)
            F4=BoundedFloatText(value=s_v[p_index],min=-100,max=100)
            F5=BoundedFloatText(value=a_min[p_index],min=-100,max=100)#下限値
            F6=BoundedFloatText(value=e_min[p_index],min=-100,max=100)
            F7=BoundedFloatText(value=r_min[p_index],min=-100,max=100)
            F8=BoundedFloatText(value=s_min[p_index],min=-100,max=100)
            F9=BoundedFloatText(value=a_max[p_index],min=-100,max=100)#上限値
            F10=BoundedFloatText(value=e_max[p_index],min=-100,max=100)
            F11=BoundedFloatText(value=r_max[p_index],min=-100,max=100)
            F12=BoundedFloatText(value=s_max[p_index],min=-100,max=100)
            C1=Checkbox(value=a_va[p_index])#フィッティングに使う変数？
            C2=Checkbox(value=e_va[p_index])
            C3=Checkbox(value=r_va[p_index])
            C4=Checkbox(value=s_va[p_index])
            T5=Text(value=feff_l[p_index])#feffXXXX.datのパス
            L1=Text(value='Parameter name')
            L2=Text(value='Initial value')
            L3=Text(value='Minimum')
            L4=Text(value='Maximum')
            L5=Text(value='Variable?')
            L6=Label('feffXXXX.dat file path')
            h1=HBox([T1,F1,F5,F9,C1])
            h2=HBox([T2,F2,F6,F10,C2])
            h3=HBox([T3,F3,F7,F11,C3])
            h4=HBox([T4,F4,F8,F12,C4])
            h5=HBox([L1,L2,L3,L4,L5])
            ui2_l=[h5,h1,h2,h3,h4]
            T6=Text(value=c3_n[p_index],disabled=not u_c3)
            F13=BoundedFloatText(value=c3_v[p_index],min=-100,max=100,disabled=not u_c3)
            F14=BoundedFloatText(value=c3_min[p_index],min=-100,max=100,disabled=not u_c3)
            F15=BoundedFloatText(value=c3_max[p_index],min=-100,max=100,disabled=not u_c3)
            C8=Checkbox(value=c3_va[p_index],disabled=not u_c3)
            h6=HBox([T6,F13,F14,F15,C8])
            ui2_l.append(h6)
            T7=Text(value=c4_n[p_index],disabled=not u_c4)
            F16=BoundedFloatText(value=c4_v[p_index],min=-100,max=100,disabled=not u_c4)
            F17=BoundedFloatText(value=c4_min[p_index],min=-100,max=100,disabled=not u_c4)
            F18=BoundedFloatText(value=c4_max[p_index],min=-100,max=100,disabled=not u_c4)
            C9=Checkbox(value=c4_va[p_index],disabled=not u_c4)
            h7=HBox([T7,F16,F17,F18,C9])
            ui2_l.append(h7)
            T8=Text(value=ei_n[p_index],disabled=not u_ei)
            F19=BoundedFloatText(value=ei_v[p_index],min=-100,max=100,disabled=not u_ei)
            F20=BoundedFloatText(value=ei_min[p_index],min=-100,max=100,disabled=not u_ei)
            F21=BoundedFloatText(value=ei_max[p_index],min=-100,max=100,disabled=not u_ei)
            C10=Checkbox(value=ei_va[p_index],disabled=not u_ei)
            h8=HBox([T8,F19,F20,F21,C10])
            ui2_l.append(h8)
            h9=HBox([L6,T5])
            ui2_l.append(h9)
            ui2=VBox(ui2_l)
            out2=interactive_output(path_param_set,{'a_value':F1,'e_value':F2,'r_value':F3,'s_value':F4,
                                                    'a_minimum':F5,'e_minimum':F6,'r_minimum':F7,'s_minimum':F8,
                                                    'a_maximum':F9,'e_maximum':F10,'r_maximum':F11,'s_maximum':F12,
                                                    'a_variable':C1,'e_variable':C2,'r_variable':C3,'s_variable':C4,
                                                    'a_name':T1,'e_name':T2,'r_name':T3,'s_name':T4,'feffdat_path':T5,
                                                    'c3_value':F13,'c3_minimum':F14,'c3_maximum':F15,'c3_variable':C8,
                                                    'c4_value':F16,'c4_minimum':F17,'c4_maximum':F18,'c4_variable':C9,
                                                    'ei_value':F19,'ei_minimum':F20,'ei_maximum':F21,'ei_variable':C10,
                                                    'c3_name':T6,'c4_name':T7,'ei_name':T8})
            display(ui2,out2)
        I1=BoundedIntText(value=No_path,min=1,max=5,description='No. of Paths')
        D1=Dropdown(value=p_list[0],options=p_list)
        C5=Checkbox(value=para2_u_l[0],description='C3')
        C6=Checkbox(value=para2_u_l[1],description='C4')
        C7=Checkbox(value=para2_u_l[2],description='Ei')
        ui=HBox([I1,D1,C5,C6,C7])
        out=interactive_output(path_set,{'No_path':I1,'pathname':D1,'u_c3':C5,'u_c4':C6,'u_ei':C7})
        display(ui,out)
        # for threading of saving result files with numpy.savetxt() added by Fujikawa on 2022.11.15
def _thread_savetxt(file_path, np_array, header_text):
    ny.savetxt(file_path, np_array, header=header_text, delimiter='\t')

#リストにある全データの解析
# modified by Fujikawa on 2022.Nov
class fit_all:
    def __init__(self):
        def on_click_ana(clicked_button: Button) -> None:
            logging.info(f' fit_all logger start on {log_date.strftime("%Y%m%d%H%M%S")}')
            logging.info(f"Initial setting values: THREAD_FT={THREAD_FT}, BINARY_SAVE={BINARY_SAVE}, THREAD_SAVE={THREAD_SAVE}, OPTIMIZE_FIT={OPTIMIZE_FIT}, THREAD_FIT={THREAD_FIT},  MEMORY_ASSESSEMENT={MEMORY_ASSESSMENT}, PROFILER={PROFILER}, BENCH={BENCH}, max_workers = {THREAD_CPU_CORE}, max_workers of XAFSana_all = {THREAD_FT_CPU_CORE}")
            logging.info(f'Initial values:\n  ipreE1={ipreE1}, ipreE2={ipreE2}, ipostE1={ipostE1}, ipostE2={ipostE2}, inormO={inormO}, irbkg1={irbkg1}, ikmin1={ikmin1}, ikmax1={ikmax1}, ikmin2={ikmin2}, ikmax2={ikmax2}, irmin1={irmin1}, irmax1={irmax1}, ikw={ikw}, F_initial={F_initial}')
            logging.info(f'loger fit_all')
            logging.info(f"Setting values: THREAD_FT={THREAD_FT}, data save={selection_data_save.value}, OPTIMIZE_FIT={checkbox_optimize_fit.value}, \nTHREAD_FIT={checkbox_thread_fit.value}, FT CPU core = {inttext_thread_core_ft.value}, fit and save CPU core = {inttext_thread_core.value}, MEMORY_ASSESSEMENT={MEMORY_ASSESSMENT}, PROFILER={checkbox_profiler.value}, BENCH={BENCH}")

            if BENCH == True:
                start_time = time.perf_counter() # for benchmark on 2202.10.25
                with open('bench.txt',mode='a+') as bench_file: # for benchmark on 2202.10.25
                    bench_file.write(f'on_click_ana on {datetime.datetime.now()} \n') # for benchmark on 2202.10.25
#            if PROFILER == True:
            if  checkbox_profiler.value == True:
                profiler_date = datetime.datetime.now()
                fitall_profiler=cProfile.Profile() # for profiler on 2202.10.26
                fitall_profiler.enable() # for profiler on 2202.10.26
            filelist,filenamelist,savefileD = read_arg1()
            savefileD = Path(savefileD) # transfer str to Path strings 
            logging.info(f'Total number of files = {len(filelist)}') 
            if savefileD.is_dir() == False:
                savefileD.mkdir()
            save_args()
            ft_cpu_core = inttext_thread_core_ft.value
#            d=XAFSana_all()
            d=XAFSana_all(ft_cpu_core )
            if C1.value==True:
                save_directory =  savefileD / 'norm' 
                if BENCH == True:  # for benchmark on 2202.10.25
                    start_time_c1 = time.perf_counter()
                if save_directory.is_dir() == False:
                    save_directory.mkdir()
                HEADER_NORM = 'energy\tnorm\tmu\tpre_edge\tpost_edge\tbkg' 
#                if THREAD_SAVE == True:  # for threading on 2202.11.15
                if selection_data_save.value == 'TEXT': # for threading on 2202.11.15
                    with concurrent.futures.ThreadPoolExecutor(max_workers=inttext_thread_core.value) as executor:
                        future_thread_savetxt = {executor.submit(_thread_savetxt, save_directory / f'norm_{fn}', ny.stack([g.e,g.flat,g.xmu,g.pre_edge,g.post_edge,g.bkg],1), HEADER_NORM ):(g, fn) for g, fn in zip(d, filenamelist)} 
#                elif BINARY_SAVE == True:  # for save files as binary files on 2202.11.17
                elif selection_data_save.value == 'BINARY': # for threading on 2202.11.15
                    binarize_data= {}
                    header_list =[]
                    for g, fn in zip(d, filenamelist):
                        binarize_data[f'norm_{fn}'] = ny.stack([g.e,g.flat,g.xmu,g.pre_edge,g.post_edge,g.bkg],1)
                        header_list.append(HEADER_NORM) # 解凍処理の時に統一化するため少し無駄な処理
                    ny.savez(save_directory / f'norm_results.npz', **binarize_data)        
                    ny.save(save_directory / f'norm_header.npy', header_list) 
                else:
                    for g, fn in zip(d, filenamelist):
                        ny.savetxt(save_directory / f'norm_{fn}', ny.stack([g.e,g.flat,g.xmu,g.pre_edge,g.post_edge,g.bkg],1),\
                        #ny.savetxt(savefileD+'/norm/norm_'+fn, ny.stack([g.e,g.flat,g.xmu,g.pre_edge,g.post_edge,g.bkg],1),\
                                   header=HEADER_NORM, delimiter='\t')
                if BENCH == True:
                    end_time = time.perf_counter() # for benchmark on 2202.10.25
                    differential_time = end_time - start_time 
                    differential_time_c1 = end_time - start_time_c1
                    with open('bench.txt',mode='a+') as bench_file:
                        bench_file.write(f'on_clic_ana and C1 are called in {differential_time}, {differential_time_c1}  on {datetime.datetime.now()} \n') 
            if C2.value==True:
                save_directory =  savefileD / 'chik' 
                if BENCH == True: # for benchmark on 2202.10.25
                    start_time_c2 = time.perf_counter()
                if save_directory.is_dir() == False:
                    save_directory.mkdir()
#                if THREAD_SAVE == True: # for threading on 2202.11.15
                if selection_data_save.value == 'TEXT': # for threading on 2202.11.15
                    with concurrent.futures.ThreadPoolExecutor(max_workers=inttext_thread_core.value) as executor:
                        future_thread_savetxt = {executor.submit(_thread_savetxt, save_directory / f'chik_{fn}',\
                                                                 ny.stack([g.k,g.chi],1), 'k\tchi'):(g, fn) for g, fn in zip(d, filenamelist)}  
#                elif BINARY_SAVE == True:  # for save files as binary files on 2202.11.17
                elif selection_data_save.value == 'BINARY': # for threading on 2202.11.15
                    binarize_data= {}
                    header_list =[]
                    for g, fn in zip(d, filenamelist):
                        binarize_data[f'chik_{fn}'] = ny.stack([g.k,g.chi],1)
                        header_list.append('k\tchi') 
                    ny.savez(save_directory / f'chik_results.npz', **binarize_data)        
                    ny.save(save_directory / f'chik_header.npy', header_list) 
                else:
                    for g, fn in zip(d, filenamelist):
                        ny.savetxt(save_directory / f'chik_{fn}', ny.stack([g.k,g.chi],1),header='k\tchi',delimiter='\t')
                        #ny.savetxt(savefileD + '/chik/chik_'+fn, ny.stack([g.k,g.chi],1),header='k\tchi',delimiter='\t')
                print('chik finished\n')
                if BENCH == True: # for benchmark on 2202.10.25
                    end_time = time.perf_counter()
                    differential_time = end_time - start_time 
                    differential_time_c2 = end_time - start_time_c2
                    with open('bench.txt',mode='a+') as bench_file:
                        bench_file.write(f'on_clic_ana and C2 are called in {differential_time}, {differential_time_c2}  on {datetime.datetime.now()} \n') 
            if C3.value==True:
                save_directory =  savefileD / 'FT' 
                if BENCH == True:
                    start_time_c3 = time.perf_counter() # for benchmark on 2202.10.25
                if save_directory.is_dir() == False:
                    save_directory.mkdir()
#                if THREAD_SAVE == True: # for threading on 2202.11.15
                if selection_data_save.value == 'TEXT': # for threading on 2202.11.15
                    with concurrent.futures.ThreadPoolExecutor(max_workers=inttext_thread_core.value) as executor:
                        future_thread_savetxt = {executor.submit(_thread_savetxt, save_directory / f'FT_{fn}', ny.stack([g.r,g.chir_mag,g.chir_re,g.chir_im],1),\
                                            f'FT k-weight={str(g.xftf_details.call_args.get("kweight"))}\nr\tchir_mag\tchir_re\tchir_im'):\
                                            (g, fn) for g, fn in zip(d, filenamelist)} 
#                elif BINARY_SAVE == True:  # for save files as binary files on 2202.11.17
                elif selection_data_save.value == 'BINARY': # for threading on 2202.11.15
                    binarize_data= {}
                    header_list =[]
                    for g, fn in zip(d, filenamelist):
                        binarize_data[f'FT_{fn}'] = ny.stack([g.r,g.chir_mag,g.chir_re,g.chir_im],1)
                        header_list.append(f'FT k-weight={str(g.xftf_details.call_args.get("kweight"))}\nr\tchir_mag\tchir_re\tchir_im')
                    ny.savez(save_directory / f'FT_results.npz', **binarize_data)        
                    ny.save(save_directory / f'FT_header.npy', header_list) 
                else:    
                    for g, fn in zip(d, filenamelist):
                        ny.savetxt(save_directory / f'FT_{fn}',  ny.stack([g.r,g.chir_mag,g.chir_re,g.chir_im],1),\
                               header='FT k-weight='+str(g.xftf_details.call_args.get("kweight"))+'\nr\tchir_mag\tchir_re\tchir_im' ,delimiter='\t')
                print('FT finished\n')
                if BENCH == True: # for benchmark on 2202.10.25
                    end_time = time.perf_counter() 
                    differential_time = end_time - start_time 
                    differential_time_c3 = end_time - start_time_c3
                    with open('bench.txt',mode='a+') as bench_file: 
                        bench_file.write(f'on_clic_ana and C3 are called in {differential_time}, {differential_time_c3}  on {datetime.datetime.now()} \n') 
            if C4.value==True:
                save_directory =  savefileD / 'Fit' 
                if BENCH == True: # for benchmark on 2202.10.25
                    start_time_c4 = time.perf_counter() # for benchmark on 2202.10.25
                d, out_l,dset_l= FEFFfit(d, checkbox_thread_fit.value, checkbox_optimize_fit.value, inttext_thread_core.value)
                if save_directory.is_dir() == False:
                    save_directory.mkdir()
                # THREAD_SAVE, THREAD_FITを併用の場合はC4のFitの部分だけtnumpy.savetxtを行わないほうが処理速度が速くなるようだ
#                if THREAD_SAVE == True != checkbox_thread_fit.value : # for threading on 2202.11.15
                if selection_data_save.value == 'TEXT': # for threading on 2202.11.15
                    with concurrent.futures.ThreadPoolExecutor(max_workers=inttext_thread_core.value) as executor:
                        future_thread_savetxt = {executor.submit(_thread_savetxt, save_directory /f'Fit_{fn}',\
                                            ny.stack([dset.data.r,dset.data.chir_mag,dset.model.r,dset.model.chir_mag],1),\
                                            f'FT k-weight={str(dset.transform.kweight)}\nr_data chir_mag_data r_model chir_mag_model' ):\
                                            (dset, fn) for dset, fn in zip(dset_l, filenamelist)} 
#                elif BINARY_SAVE == True:  # for save files as binary files on 2202.11.17
                elif selection_data_save.value == 'BINARY': # for threading on 2202.11.15
                    binarize_data = {}
                    header_list =[]
                    for dset, fn in zip(dset_l, filenamelist):
                        binarize_data[f'Fit_{fn}'] = ny.stack([dset.data.r,dset.data.chir_mag,dset.model.r,dset.model.chir_mag],1)
                        header_list.append(f'FT k-weight={str(dset.transform.kweight)}\nr_data chir_mag_data r_model chir_mag_model' )
                    ny.savez(save_directory / f'Fit_results.npz', **binarize_data)        
                    ny.save(save_directory / f'Fit_header.npy', header_list) 
                else: 
                    for dset, fn in zip(dset_l, filenamelist):
                        ny.savetxt(save_directory / f'Fit_{fn}', ny.stack([dset.data.r,dset.data.chir_mag,dset.model.r,dset.model.chir_mag],1),\
                               header='FT k-weight=' + str(dset.transform.kweight)+'\nr_data chir_mag_data r_model chir_mag_model',delimiter='\t')
                print('Fit finished\n')
                if BENCH == True: # for benchmark on 2202.10.25
                    end_time = time.perf_counter() 
                    differential_time = end_time - start_time
                    differential_time_c4 = end_time - start_time_c4
                    with open('bench.txt',mode='a+') as bench_file:
                        bench_file.write(f'on_clic_ana and C4 are called in {differential_time}, {differential_time_c4}  on {datetime.datetime.now()} \n') 
            if C5.value==True:
                if BENCH==True:
                    start_time_c5 = time.perf_counter() # for benchmark on 2202.10.25
                if C4.value==False:
                    d,out_l,dset_l= FEFFfit(d, checkbox_thread_fit.value, checkbox_optimize_fit.value, inttext_thread_core.value)
                No_path=read_arg3()[0]
                HEADER_RESULTS='s02\te0\tdeltar\tsigma2\tthird\tfourth\tei\ts02_stderr\te0_stderr\tdeltar_stderr\tsigma2_stderr\tthird_stderr\tfourth_stderr\tei_stderr'
                for j in range(No_path):
                    #ny.savetxt(savefileD+'/fitting_results_path'+str(j+1)+'.dat',\
                    ny.savetxt(savefileD/ f'fitting_results_path{str(j+1)}.dat',\
                                ny.stack([out_l[j,:,0],out_l[j,:,1],out_l[j,:,2],out_l[j,:,3],out_l[j,:,9],\
                                out_l[j,:,10],out_l[j,:,11], out_l[j,:,5],out_l[j,:,6],out_l[j,:,7],\
                                out_l[j,:,8],out_l[j,:,12],out_l[j,:,13],out_l[j,:,14]],1),\
                                header=HEADER_RESULTS, delimiter='\t')
                print('Output results file finished\n')
                if BENCH==True: # for benchmark on 2202.10.25
                    end_time = time.perf_counter()
                    differential_time = end_time - start_time
                    differential_time_c5 = end_time - start_time_c5
                    with open('bench.txt',mode='a+') as bench_file:
                        bench_file.write(f'on_clic_ana and C5 are called in {differential_time}, {differential_time_c5}  on {datetime.datetime.now()} \n') 
#            if PROFILER==True:
            if  checkbox_profiler.value == True:
                fitall_profiler.disable() # for profiler on 2202.10.26
                with open(f'fitall{profiler_date.strftime("%Y%m%d%H%M")}.prof',mode='a+') as fitall_profile:
                    sortkey = SortKey.CUMULATIVE
                    ps = pstats.Stats(fitall_profiler, stream=fitall_profile).sort_stats(sortkey)
                    ps.print_stats()
                    sortkey = SortKey.TIME 
                    ps = pstats.Stats(fitall_profiler, stream=fitall_profile).sort_stats(sortkey)
                    ps.print_callers(.5, 'init')
                    ps.print_stats(20)
            if MEMORY_ASSESSMENT == True: # for momory monitor on 2022.11.17
                import sys
                print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))
                print(" ------------------------------------ ")
                for var_name in dir():
                    if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > 0: #ここだけアレンジ
                        print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))
        C1=Checkbox(value=False,description='norm')
        C2=Checkbox(value=False,description='chik')
        C3=Checkbox(value=False,description='FT')
        C4=Checkbox(value=False,description='Fit')
        C5=Checkbox(value=True,description='Fitting results')
        B1=Button(description='start')
        B1.on_click(on_click_ana)
        h1=HBox([C1,C2,C3,C4,C5])
        ui=VBox([h1,B1])
        display(ui)
        
        # 評価用チェックボックス
        layout_object=Layout(width='400px', height='40px')
        layout_box=Layout(justify_content='flex-start')
        cp_value = PROFILER
        checkbox_profiler = Checkbox(
                value=cp_value,
                description='PROFILER: ',
                disabled=False,
                layout=layout_object
            )
        ctf_value = THREAD_FIT
        checkbox_thread_fit = Checkbox(
                value=ctf_value ,
                description='THREAD_FIT: ',
                disabled=False,
                layout=layout_object
            )
        cof_value = OPTIMIZE_FIT
        checkbox_optimize_fit = Checkbox(
                value=cof_value,
                description='OPTIMIZE_FIT: ',
                disabled=False,
                layout=layout_object
            )
        itf_value = THREAD_FT_CPU_CORE
        inttext_thread_core_ft = IntText(
                value=itf_value ,
                description='THREAD_FT max_workers:',
                disabled=False,
                layout=layout_object
            )
        it_value = THREAD_CPU_CORE
        inttext_thread_core = IntText(
                value=it_value ,
                description='THREAD Fit,save max_workers:',
                disabled=False,
                layout=layout_object
            )
        if BINARY_SAVE is True:
            data_save_value = 'BINARY'
        elif THREAD_SAVE is True:
            data_save_value = 'TEXT'
        selection_data_save = Select(
                options=['BINARY','TEXT'],
                value=data_save_value,
                disable=False,
                description='SAVE:',
                layout=layout_object
            )
        inttext_thread_core_ft.style.description_width='200px'
        inttext_thread_core.style.description_width='200px'
        hbox_upper = HBox([checkbox_profiler, checkbox_thread_fit, checkbox_optimize_fit])
        hbox_bottoms = HBox([inttext_thread_core_ft, inttext_thread_core, selection_data_save])
        vbox_options = VBox([hbox_upper, hbox_bottoms], layout=layout_box)
        display(vbox_options)
        
# fitting resultのグラフ描画
class res_plot:
    def __init__(self):
        self.fig2=plt.figure()
        def res_plotter(select,select2,dx,path):
            self.fig2.clear()
            f_list,f_name_list,save_f_D=read_arg1()
            res_list=glob.glob(save_f_D + '/fitting_results_path*.dat')
            if len(res_list) > path:
                data=ny.loadtxt(res_list[path], unpack=True)
                ax1=self.fig2.add_subplot(111)
                x=ny.arange(len(data[:][0]))*dx
                plotdata=data[:][select]
                ploterror=data[:][select+7]
                ax1.plot(x, plotdata, color = 'b')
                if select2 != 7:
                    plotdata2=data[:][select2]
                    ax2=ax1.twinx()
                    ax2.plot(x, plotdata2, color = 'r')
        S1=Dropdown(options=[('S02',0), ('enot',1),('deltar',2),('sigma2',3), ('third',4), ('forth',5), ('ei',6)], value=2, description='left axis')
        S2=Dropdown(options=[('S02',0), ('enot',1),('deltar',2),('sigma2',3), ('third',4), ('forth',5), ('ei',6), ('none',7)], value=7, description='right axis')
        S3=Dropdown(options=[('path1',0), ('path2',1), ('path3', 2), ('path4', 3), ('path5', 4)], value = 0)
        F1=BoundedFloatText(value=1,min=0.000001, description='delta x')
        ui=HBox([S3,S1,S2,F1])
        out = interactive_output(res_plotter, {'select':S1, 'select2':S2, 'dx':F1, 'path':S3})
        display(ui,out)

# データnpzファイルとヘッダーnpyファイルの解凍 
def unzip_data_pane():
    unzip_button = Button(
       description='Start',
       disabled=False,
       button_style='', # 'success', 'info', 'warning', 'danger' or ''
       tooltip='start unzip npz file',
       icon='check' # (FontAwesome names without the `fa-` prefix)
    )
    label_unzip_button = HBox([Label(value="Unzip"), unzip_button])  
    unzip_button.on_click(_unzip_data)
    display(label_unzip_button)


#@jit # numbaが使用できない
def _unzip_data(button_handler):
    # global progress_value 
    # global total_file
    FILE_SAFIX = '_results.npz'
    HEADER_SAFIX = '_header.npy'
    filelist,filenamelist,savefileD=read_arg1()
    # progress_value = 0
    # total_file = 0
    savefileD = Path(savefileD)
    directories = [savefileD / prefix for prefix in PREFIXES]
    npz_files = []
    npy_headers = []
    first_roop = True
    for directory, prefix in zip(directories, PREFIXES):
        npz_files.append(f'{directory}/{prefix}{FILE_SAFIX}')
        npy_headers.append(f'{directory}/{prefix}{HEADER_SAFIX}')
    for npz_file, directory, npy_header in zip(npz_files, directories, npy_headers):
        try:
            loaded_data = ny.load(npz_file)
            loaded_header = ny.load(npy_header)
            total_loaded_data = len(loaded_data)
            if total_loaded_data == len(loaded_header):
                _save_to_text_with_header(directory, loaded_data, loaded_header)
                print(f'{npz_file} was finished')
            else:
                print(f'list length is not match between npz_file ({total_loaded_data}) and header_list ({len(loaded_header)})')
                logging.error(f'list length is not match between npz ({total_loaded_data}) and header_list ({len(loaded_header)})')
        except FileNotFoundError:
            npzdataname = os.path.basename(npz_file)
            headername = os.path.basename(npy_header)
            print(f'Not found {npzdataname} and/or {headername} ! Place {npzdataname} and/or {headername} on {directory}')
            logging.error(f'Not found {npzdataname} and/or {headername} ! Place {npzdataname} and/or {headername} on {directory}')

@jit
def _save_to_text_with_header(directory, loaded_npz_file, header_list):
    for l_n, header_l in zip(loaded_npz_file, header_list):
        ny.savetxt(f'{directory}/{l_n}', loaded_npz_file[l_n], header=header_l, delimiter='\t')