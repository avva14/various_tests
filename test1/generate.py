import numpy as np
import cv2
import os
import uuid
from PIL import Image
from math import ceil
from chessutils import find_coeffs
import argparse

NUMCELL = 8
CELL = 45
BOARDSIZE = CELL * NUMCELL
IMGSIZE = 480
MAXSHEAR = 0.05
MINSCALE = 0.5

figs = ['p', 'b', 'n', 'r', 'q', 'k']
colors = ['d', 'l']

def boardgen(figuresimgs, distortrnd):
    '''
    figuresimgs -- dictionary of images read from 'img' folder
    distortrnd -- random array size 13
    Returns numpy array shape (IMGSIZE,IMGSIZE), list with figures positions
    '''
    blank_image = np.zeros((BOARDSIZE,BOARDSIZE), np.uint8)
    rec = []
    for i in range(NUMCELL*NUMCELL):
        xp = i % NUMCELL
        yp = i // NUMCELL

        empty = np.random.randint(4)
        fgnm = 'x'
        fgcl = 'x'
        if (empty == 0):
            figix = np.random.randint(len(figs))
            fgnm = figs[figix]
            fgcl = colors[np.random.randint(2)]
            rec.append([i, fgnm, fgcl])

        fldcol = 'l' if (xp+yp) % 2 == 0 else 'd'
        fkey = f'{fgnm}{fgcl}{fldcol}'
        blank_image[yp*CELL:(yp+1)*CELL,xp*CELL:(xp+1)*CELL] = figuresimgs[fkey]

    img = Image.fromarray(blank_image, 'L')

    m = -distortrnd[:8] * MAXSHEAR * BOARDSIZE
    x1, y1 = m[0:2]
    x2, y2 = BOARDSIZE+m[2], m[3]
    x3, y3 = BOARDSIZE+m[4], BOARDSIZE+m[5]
    x4, y4 = m[6], BOARDSIZE+m[7]

    xmin = min(x1, x4)
    if (xmin < 0):
        x1 -= xmin
        x2 -= xmin
        x3 -= xmin
        x4 -= xmin

    ymin = min(y1, y2)
    if (ymin < 0):
        y1 -= ymin
        y2 -= ymin
        y3 -= ymin
        y4 -= ymin

    greyscale = int((0.5 + distortrnd[8]*0.5) * 255)

    new_width = int(ceil(max(x2,x3,BOARDSIZE)))
    new_height = int(ceil(max(y3,y4,BOARDSIZE)))

    coeffs = find_coeffs(
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
        [(0, 0), (BOARDSIZE, 0), (BOARDSIZE, BOARDSIZE), (0, BOARDSIZE)])

    img = img.transform((new_width, new_height), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor = greyscale)

    offsets = distortrnd[9:11]
    scalex, scaley = MINSCALE + (1 - MINSCALE) * distortrnd[11:13]

    width, height = img.size
    scaled_wid, scaled_hei = int(round(width * scalex)), int(round(height * scaley))
    img = img.resize((scaled_wid, scaled_hei))

    width, height = img.size

    xoff = (IMGSIZE - width) * offsets[0]
    yoff = (IMGSIZE - height) * offsets[1]

    offset = (int(round(xoff)), int(round(yoff)))

    empty = Image.new('L', (IMGSIZE, IMGSIZE), greyscale)
    empty.paste(img, offset)

    return np.asarray(empty), rec

def distort(img, distortrnd):
    '''
    img -- PIL image
    distortrnd -- random array size 8
    Returns distorted image (random perspective)
    '''
    ns = img.size[0]
    m = (distortrnd[:8]) * 0.25 * ns

    x1, y1 = -m[0:2]
    x2, y2 = ns+m[2], m[3]

    x3, y3 = ns+m[4], ns-m[5]
    x4, y4 = -m[6], ns+m[7]

    xmin = min(x1, x4)
    if (xmin < 0):
        x1 -= xmin
        x2 -= xmin
        x3 -= xmin
        x4 -= xmin

    ymin = min(y1, y2)
    if (ymin < 0):
        y1 -= ymin
        y2 -= ymin
        y3 -= ymin
        y4 -= ymin

    new_width = int(ceil(max(x2,x3,ns)))
    new_height = int(ceil(max(y3,y4,ns)))

    coeffs = find_coeffs(
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
        [(0, 0), (ns, 0), (ns, ns), (0, ns)])
    img = img.transform((new_width, new_height), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor = 0)
    return img

def moirebackground(rndarr):
    '''
    rndarr -- random array size 8
    Returns numpy array shape (IMGSIZE,IMGSIZE) with moire pattern
    '''
    ns = 2*IMGSIZE
    grey = 128
    blank = 128 + grey*np.ones((ns,ns), np.uint8)
    for i in range(ns):
        if i % 3 != 0:
            continue
        blank[i] = grey*np.ones(ns, np.uint8)
    img = Image.fromarray(blank)

    rndxxx = rndarr*np.array([1,-1,1,-1,1,-1,1,-1])
    
    im1 = distort(img, rndarr)
    im2 = distort(img, rndxxx)
    
    xs = max(im1.size[0], im2.size[0])
    ys = max(im1.size[1], im2.size[1])
    
    im1 = im1.resize((xs,ys)) 
    im2 = im2.resize((xs,ys)) 
    
    re1 = np.asarray(im1)
    re2 = np.asarray(im2)
    
    ims = Image.fromarray(np.maximum(re1,re2))
    ims = ims.crop((IMGSIZE//2, IMGSIZE//2, IMGSIZE//2+IMGSIZE, IMGSIZE//2+IMGSIZE))
    ims = ims.resize((120,120))
    ims = ims.resize((IMGSIZE,IMGSIZE))
    
    res = np.asarray(ims) / 255
    
    maxvalshift = 1 - res.max()
    res += maxvalshift

    return res

def main():
    parser = argparse.ArgumentParser(description="Generate random images",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder',
        type=str,
        default='.'
    )

    parser.add_argument(
        '-m', '--moire',
        help='Set this flag if you want moire pttern to be added in the background.',
        action='store_true'
    )

    parser.add_argument(
        '-n', '--number',
        help='Number of images',
        default=1,
        type=int)

    args = parser.parse_args()

    folder = 'img'
    outfolder = args.outputDir

    fl = [f for f in os.listdir(folder) if f.split('_')[0] == 'Chess']
    
    figuresimgs = dict()
    for f in fl:
        fn = f.split('_')[1].split('4')[0]
        img = cv2.imread(os.path.join(folder, f))
        figuresimgs[fn] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for _ in range(args.number):

        #for a filename
        idx = uuid.uuid4() 

        #random board with figure positions
        boardimage, recs = boardgen(figuresimgs, np.random.rand(13))

        if (args.moire):
            #random moire pattern
            moir = moirebackground(np.random.rand(8))
            #combining together
            boardimage = boardimage*moir
        result = Image.fromarray(boardimage.astype(np.uint8), 'L')

        #writing image/adding record for figure positions
        result.save(os.path.join(outfolder,f'{idx}.png'))
        with open(os.path.join(outfolder, 'position.csv'), "a") as file_object:
            for b in recs:
                file_object.write(f"{idx},{b[0]},{b[1]},{b[2]}\n")    
        
    return

if __name__=='__main__':
    main()