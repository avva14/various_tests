import numpy as np
from clusterization import clusterize

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

def xcrosscorr(bd):
    imgsize = bd.shape[0]
    hfimgsize = imgsize // 2
    cftbd = np.fft.fft2(bd)

    cross = np.real(np.fft.ifft2(np.conj(cftbd)*cftbd))
    xross = np.zeros((imgsize,imgsize))

    for i in range(imgsize*imgsize):
        x = i % imgsize
        y = i // imgsize
        xs = (x - hfimgsize) % imgsize
        ys = (y - hfimgsize) % imgsize
        xross[y,x] = cross[ys,xs]
    return xross

def mhdist(x, y):
    return np.sum(np.abs(np.array(x)-np.array(y)))

def detectsdots(bd):
    
    invbd = 1 - bd

    #Clusterizing points
    xsumms = np.sum(invbd, axis=-1)
    res = []
    for y, xsumm in enumerate(xsumms):
        if xsumm == 0:
            continue
        ##TODO extend with zeros??
        s = invbd[y]
        diffs = s[1:]-s[:-1]
        starts = np.where(diffs==1)[0]
        ends = np.where(diffs==-1)[0]
        for i1, i2 in zip(starts,ends):
            res.append([i1+1,i2,y])
    cl = clusterize(res, closeintervals)
    return np.array(list(map(centerofmass, cl)))

def detectshifts(bd):
    xc = xcrosscorr(bd)
    imgsize = bd.shape[0]
    hfimgsize = imgsize // 2
    xp1 = np.zeros_like(xc)
    xp2 = np.zeros_like(xc)
    xp1[:,hfimgsize-10:hfimgsize+10] = xc[:,hfimgsize-10:hfimgsize+10]
    xp1[hfimgsize-10:hfimgsize+10,:] = 0
    xp2[hfimgsize-10:hfimgsize+10,:] = xc[hfimgsize-10:hfimgsize+10,:]
    xp2[:,hfimgsize-10:hfimgsize+10] = 0

    p1 = np.argmax(xp1)
    p2 = np.argmax(xp2)
    yshift = np.array([p1 % imgsize - hfimgsize, p1 // imgsize - hfimgsize])
    xshift = np.array([p2 % imgsize - hfimgsize, p2 // imgsize - hfimgsize])
    
    if xshift[0] < 0:
        xshift *= -1
    if yshift[1] < 0:
        yshift *= -1
    return np.array([xshift,yshift])

def detectnei(purecl, vect, start=0):
    abdir = np.vstack((np.eye(2, dtype=int),-np.eye(2, dtype=int)))
    visited = np.zeros(len(purecl)).astype(bool)
    ixysites = np.zeros((len(purecl),2), dtype=int)
    checked = np.zeros(len(purecl)).astype(bool)
    visited[start] = True

    while (np.sum(visited) < len(purecl)):
        nvisited = np.sum(visited)
        for ns in range(len(purecl)):

            if checked[ns]:
                continue
            if not(visited[ns]):
                continue
            p = purecl[ns]

            for ab in abdir:
                pxy = p + np.dot(ab, vect)
                xyd = {i:mhdist(c,pxy) for i,c in enumerate(purecl)}
                xydm = min(xyd, key=xyd.get)
                #TODO check visited[xydm]
                if (xyd[xydm] < 10):
                    visited[xydm] = True
                    ixysites[xydm] = ixysites[ns] + ab
            checked[ns] = True
        if (np.sum(visited) == nvisited):
            break
            
    return purecl[visited], ixysites[visited]

def xypos(xx):
    if (xx[0] >= 0) and (xx[0] <= 6) and (xx[1] >= 0) and (xx[1] <= 6):
        return 7 * xx[1] + xx[0]
def posxy(xx):
    return 7 * xx[1] + xx[0]
def neinei(ix):
    return [ix % 7, ix // 7]

def detectcorners(vec, pts, ixdict):
    abdir = np.vstack((np.eye(2, dtype=int),-np.eye(2, dtype=int)))
    cornerixes = [0, 6, 42, 48]
    res = np.zeros((4,2))

    for num, ix in enumerate(cornerixes):

        if ix in ixdict: 
            res[num] = pts[ixdict[ix]]
        else:

            neiix = []
            xyp = neinei(ix)
            for ab in abdir:
                s = xyp + ab
                z = xypos(s)
                if (z in ixdict):
                    neiix.append(pts[ixdict[z]] - vec@ab)
            res[num] = np.mean(neiix, axis=0)

    return res

def cornersdetection(bd):
    xyshift = detectshifts(bd)
    xy = detectsdots(bd)

    npt = 0
    istart = 0
    while (npt < xy.shape[0] // 2):
        pts, ix = detectnei(xy, xyshift, istart)
        istart += 1
        npt = pts.shape[0]

    minc = np.min(ix, axis=0)
    maxc = np.max(ix, axis=0)
    
    minxpos = minc[0]
    if (maxc[0] - minc[0] < 6):
        return None
    elif (maxc[0] - minc[0] > 6):
        xc = np.unique(ix[:,0], return_counts=True)
        minxpos = np.min([val for val, cnt in zip(xc[0], xc[1]) if cnt > 3])
        
    minypos = minc[1]
    if (maxc[1] - minc[1] < 6):
        return None
    elif (maxc[0] - minc[0] > 6):
        yc = np.unique(ix[:,1], return_counts=True)
        minypos = np.min([val for val, cnt in zip(yc[0], yc[1]) if cnt > 3])
    
    minix = posxy([minxpos, minypos])
    ixdict = {posxy(p)-minix:i for i, p in enumerate(ix)}

    return detectcorners(xyshift, pts, ixdict)
