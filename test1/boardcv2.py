import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from chessutils import *
import argparse

CELL = 45
NUMCELL = 8
BOARDSIZE = CELL * NUMCELL

def figdetectcv2(figuresimgs, inputimg, threshld=0):
    '''
    Detection routine
    '''

    ncorner = 7
    prs = cv2.findChessboardCorners(inputimg, (ncorner,ncorner))
    if not prs[0]:
        return

    xy = np.array([t[0] for t in prs[1]])
    xyc = np.mean(xy, axis=0)
    mhdists = [np.sum(np.abs(p-xyc)) for p in xy]
    corners = xy[np.argsort(mhdists)[-4:]]

    half = 0.5*np.max(mhdists)
    newc = xyc + half*np.array([[-1,-1],[1,-1],[-1,1],[1,1]])

    ixmatch = [np.argmin(np.sum(np.abs(p-corners), axis=1)) for p in newc]
    cornersmatch = corners[ixmatch]

    brdsize = half / (ncorner-1) * 8

    coeffs = find_coeffs(newc, cornersmatch)
    img = Image.fromarray(inputimg)
    img = img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    img = img.crop((xyc[0]-brdsize, xyc[1]-brdsize, xyc[0]+brdsize, xyc[1]+brdsize))
    img = img.resize((BOARDSIZE,BOARDSIZE))
    cropped = np.asarray(img)

    outdict = dict()

    for i in range(NUMCELL*NUMCELL):
        xp = i % NUMCELL
        yp = i // NUMCELL
        
        xt = cropped[CELL*yp:CELL*(yp+1),CELL*xp:CELL*(xp+1)]
        cl = 'l' if (xp+yp) % 2 == 0 else 'd'
        
        vc = threshld
        ky = 'xx'
        for k, v in figuresimgs.items():
            if (k[-1] != cl):
                continue
            res = cv2.matchTemplate(xt, v, cv2.TM_CCOEFF_NORMED)
            maxpos = np.argmax(res)
            if (abs(maxpos%res.shape[0]-10.5) > 5 or abs(maxpos//res.shape[0]-10.5) > 5):
                continue
            if (np.ravel(res)[maxpos] > vc):
                vc = np.ravel(res)[maxpos]
                ky = k[:2]
        if (ky != 'xx'):
            outdict[i] = ky
    return outdict


def main():

    parser = argparse.ArgumentParser(description="Generate random images",
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

    fl = [os.path.join('img', f) for f in os.listdir('img') if f.split('_')[0] == 'Chess']
    bkgray = dict()
    for f in fl:
        if (f.split('4')[0][-2] != 'x'):
            continue
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bkgray[f.split('_')[1].split('4')[0][-1]] = img[0][0]

    figuresimgs = dict()
    for f in fl:
        fn = f.split('_')[1].split('4')[0]
        bk = fn[-1]
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        figuresimgs[fn] = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=float(bkgray[bk]))

    if (args.validation):
        valfile = os.path.join(args.dir, 'position.csv')
        assert os.path.isfile(valfile), "No validation file"
        df = pd.read_csv(valfile, header=None)
        df.columns = ['filename','pos','figure','color']

    flist = [f for f in os.listdir(args.dir) if f.split('.')[-1] == 'png']
    for f in flist:
        img = cv2.imread(os.path.join(args.dir, f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pred = figdetectcv2(figuresimgs, gray, 0.5)

        print(f)
        if (args.validation):
            validation = df.loc[df['filename'] == f.split('.')[0]]
            validdict = {rec.pos:f'{rec.figure}{rec.color}' for _, rec in validation.iterrows()}
            outres(pred, validdict)
        else:
            outpred(pred)

    return

if __name__=='__main__':
    main()