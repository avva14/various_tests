import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from PIL import Image
from clusterization import clusterize
from chessutils import *
import argparse

NUMCELL = 8
IMGSIZE = 480
SIZE = IMGSIZE // NUMCELL

model = tf.keras.models.load_model(os.path.join('models','unet_board_v4.h5'))
classifier = tf.keras.models.load_model(os.path.join('models','class_figure_col_v4.h5'), compile=False)

colors = np.array([' ', 'd', 'l'])
figures = np.array([' ','p','b','n','r','q','k'])

def intersect(i1, i2):
    return (i2[0] <= i1[1]) and (i1[0] <= i2[1])
def closeintervals(i1, i2):
    return intersect(i1, i2) and (abs(i1[2]-i2[2]) == 1)

def centerofmass(c):
    xm = 0
    ym = 0
    nm = 0
    for i in c:
        ns = i[1]-i[0]+1
        xm += 0.5*(i[1]+i[0])*ns
        ym += i[2]*ns
        nm += ns
    return xm/nm, ym/nm

def figdetectcnn(boardmodel, classmodel, inputimg):
    '''
    Detection routine
    '''
    img = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMGSIZE,IMGSIZE))
    feed = np.expand_dims(img/255, axis=(0,-1))
    pred = np.squeeze(boardmodel.predict(feed))
    bp = 1 - np.argmax(pred, axis=-1)

    xsumms = np.sum(bp, axis=-1)
    res = []
    for y, xsumm in enumerate(xsumms):
        if xsumm == 0:
            continue
        s = bp[y]
        diffs = s[1:]-s[:-1]
        starts = np.where(diffs==1)[0]
        ends = np.where(diffs==-1)[0]
        for i1, i2 in zip(starts,ends):
            res.append([i1+1,i2,y])

    cl = clusterize(res, closeintervals)

    xy = np.array(list(map(centerofmass, cl)))
    xyc = np.mean(xy, axis=0)
    mhdists = [np.sum(np.abs(p-xyc)) for p in xy]
    corners = xy[np.argsort(mhdists)[-4:]]

    half = 0.5 * np.max(mhdists)
    newc = xyc + half*np.array([[-1,-1],[1,-1],[-1,1],[1,1]])

    ixmatch = [np.argmin(np.sum(np.abs(p-corners), axis=1)) for p in newc]
    cornersmatch = corners[ixmatch]
    brdsize = half / 6 * NUMCELL
 
    coeffs = find_coeffs(newc, cornersmatch)

    img = Image.fromarray(img)
    img = img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    img = img.crop((xyc[0]-brdsize, xyc[1]-brdsize, xyc[0]+brdsize, xyc[1]+brdsize))

    img = img.resize((IMGSIZE,IMGSIZE))

    cropped = np.asarray(img)
    stacked = np.zeros((NUMCELL*NUMCELL,SIZE,SIZE))
    for i in range(NUMCELL*NUMCELL):
        xp = i % NUMCELL
        yp = i // NUMCELL
        stacked[i] = cropped[yp*SIZE:(yp+1)*SIZE,xp*SIZE:(xp+1)*SIZE]/255
    stacked = np.expand_dims(stacked,axis=-1)

    preds = classmodel.predict(stacked)
    a1 = figures[np.argmax(preds[:,0:7],axis=-1)]
    a2 = colors[np.argmax(preds[:,7:10],axis=-1)]

    outdict = {i:f'{t1}{t2}' for i, (t1,t2) in enumerate(zip(a1, a2)) if t1 != ' ' and t2 != ' '}
    return outdict

def main():

    parser = argparse.ArgumentParser(description="Detect figures positions",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-d', '--dir',
        help='Path to the folder',
        type=str,
        default='.'
    )

    parser.add_argument(
        '-v', '--validation',
        help='Set this flag if position.csv file exists',
        action='store_true'
    )

    args = parser.parse_args()

    if (args.validation):
        valfile = os.path.join(args.dir, 'position.csv')
        assert os.path.isfile(valfile), "No validation file"
        df = pd.read_csv(valfile, header=None)
        df.columns = ['filename','pos','figure','color']

    flist = [f for f in os.listdir(args.dir) if f.split('.')[-1] == 'png']
    for f in flist:

        img = cv2.imread(os.path.join(args.dir, f))
        pred = figdetectcnn(model, classifier, img)

        print(f)
        if (args.validation):
            validation = df.loc[df['filename'] == f.split('.')[0]]
            validdict = {rec.pos:f'{rec.figure}{rec.color}' for _, rec in validation.iterrows()}
            outres(pred, validdict)
        else:
            outpred(pred)
    
if __name__=='__main__':
    main()