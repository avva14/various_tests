import numpy as np
from math import ceil
from PIL import Image
from chessutils import find_coeffs

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

def moirebackground(rndarr, imgsize):
    '''
    rndarr -- random array size 8
    Returns numpy array shape (IMGSIZE,IMGSIZE) with moire pattern
    '''
    ns = 2*imgsize
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
    ims = ims.crop((imgsize//2, imgsize//2, imgsize//2+imgsize, imgsize//2+imgsize))
    ims = ims.resize((120,120))
    ims = ims.resize((imgsize,imgsize))
    
    res = np.asarray(ims) / 255
    
    maxvalshift = 1 - res.max()
    res += maxvalshift

    return res

def chessboard(figuresimgs, distortrnd, imgsize, params):
    '''
    figuresimgs -- dictionary of images read from 'img' folder
    distortrnd -- random array size 13
    Returns numpy array shape (imgsize,imgsize), list with figures positions
    '''
    numcell = params['numcell']
    cell = params['cellsize']
    figs = params['figures']
    colors = params['colors']
    maxshear = params['shear']
    minscale = params['scale']

    boardsize = cell * numcell

    blank_image = np.zeros((boardsize,boardsize), np.uint8)
    rec = []
    for i in range(numcell*numcell):
        xp = i % numcell
        yp = i // numcell

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
        blank_image[yp*cell:(yp+1)*cell,xp*cell:(xp+1)*cell] = figuresimgs[fkey]

    img = Image.fromarray(blank_image, 'L')

    m = -distortrnd[:8] * maxshear * boardsize
    x1, y1 = m[0:2]
    x2, y2 = boardsize+m[2], m[3]
    x3, y3 = boardsize+m[4], boardsize+m[5]
    x4, y4 = m[6], boardsize+m[7]

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

    new_width = int(ceil(max(x2,x3,boardsize)))
    new_height = int(ceil(max(y3,y4,boardsize)))

    coeffs = find_coeffs(
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
        [(0, 0), (boardsize, 0), (boardsize, boardsize), (0, boardsize)])

    img = img.transform((new_width, new_height), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor = greyscale)

    offsets = distortrnd[9:11]
    scalex, scaley = minscale + (1 - minscale) * distortrnd[11:13]

    width, height = img.size
    scaled_wid, scaled_hei = int(round(width * scalex)), int(round(height * scaley))
    img = img.resize((scaled_wid, scaled_hei))

    width, height = img.size

    xoff = (imgsize - width) * offsets[0]
    yoff = (imgsize - height) * offsets[1]

    offset = (int(round(xoff)), int(round(yoff)))

    empty = Image.new('L', (imgsize, imgsize), greyscale)
    empty.paste(img, offset)

    return np.asarray(empty), rec

