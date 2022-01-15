import numpy as np

MSIZE = 5
NSIZE = 4
SIZE = MSIZE * NSIZE
TOLERANCE = 1e-5
MAXEPOCHS = 200

def globalpos(x,y):
    '''
    Array position for x and y coordinates 
    '''
    return y * MSIZE + x

def neighgenerate():
    '''
    Forms a list with neighbor indexes
    '''
    neighbors = []
    for i in range(SIZE):
        y = i // MSIZE
        x = i % MSIZE
        nh = []
        if (x > 0):
            nh.append(globalpos(x-1,y))
        if (x < MSIZE-1):
            nh.append(globalpos(x+1,y))
        if (y > 0):
            nh.append(globalpos(x,y-1))
        if (y < NSIZE-1):
            nh.append(globalpos(x,y+1))
        neighbors.append(np.array(nh))
    return neighbors

def randomneighbour(nbs, pr):
    '''
    Picks random neighbor
    '''
    dirix = int(pr * len(nbs))
    return nbs[dirix]

def goodneighbor(nbs, rij, pr):
    '''
    Picks a neighbor corresponding maximal rij
    '''
    rneighbors = rij[nbs]
    maxps = np.argmax(rneighbors)
    rval = rneighbors[maxps]
    others = np.where(rneighbors == rval)[0]
    if (len(others) == 1):
        return nbs[maxps]
    else:
        dirix = int(pr * len(others))
        ox = others[dirix]
        return nbs[ox]

def main():

###parameters block
# number of steps in 'epoch'
    NSTEPS = 10000

# p probability value
    pprob = 0.8

#q probability value
    qprob = 0.5

# setting Rij values
    rm = np.zeros(SIZE)
    rm[10] = 10

#generates lists for nearests neighbors
    nb = neighgenerate()

#sums of Rij values for random walk
    nsite = np.zeros(SIZE)

#number of visits of ite site
    ncounter = np.zeros(SIZE, dtype=np.float32)
    nstartcounter = np.zeros(SIZE, dtype=np.float32)

#starting site
    site = np.random.randint(SIZE)
    startsite = site
    nstartcounter[startsite] = 1

    stepcounter = 1

    #File for probability on site by time
    fileprob = open('prob.bin', 'wb')
    #File for mean Rij on site by time
    filersum = open('rsum.bin', 'wb')

    prevprobs = np.zeros(SIZE, dtype=np.float32)

    for _ in range(MAXEPOCHS):

        siteprobepoch = np.zeros((NSTEPS,SIZE), dtype=np.float32)
        sitevaluepoch = np.zeros((NSTEPS,SIZE), dtype=np.float32)

        randsarray = np.random.rand(NSTEPS,3)
        for i, zz in enumerate(randsarray):
            z1, z2, z3 = zz
            if (z1 < pprob):
                ##СМЕРТЬ, рождение на случайном месте
                site = int(z3 * SIZE)
                ##Обновление счетчика
                startsite = site
                nstartcounter[startsite] += 1
            elif (z2 < qprob):
                ##Переход на соседа
                neighborsforsite = nb[site]
                site = randomneighbour(neighborsforsite, z3)
            else:
                ##Переход на благожелательного соседа
                neighborsforsite = nb[site]
                site = goodneighbor(neighborsforsite, rm, z3)

            ncounter[site] += 1
            nsite[startsite] += rm[site]
            stepcounter += 1

            siteprobepoch[i] = (ncounter / stepcounter).astype(np.float32)
            sitevaluepoch[i] = (nsite / np.maximum(nstartcounter, 1)).astype(np.float32)

        probs = np.mean(siteprobepoch, axis=0)
        if (np.max(abs(probs-prevprobs)) < TOLERANCE):
            break

        prevprobs = probs

        fileprob.write(siteprobepoch)
        filersum.write(sitevaluepoch)

    fileprob.close()
    filersum.close()

    #Final values
    result = np.zeros((3,SIZE))
    result[0] = rm
    result[1] = probs
    result[2] = np.mean(sitevaluepoch, axis=0)

    fileres = open('result.bin', 'wb')
    fileres.write(result)
    fileres.close()

    return

def mainsmooth():

###parameters block
# number of steps in 'epoch'
    NSTEPS = 1000
    NSMOOTH = 100

# p probability value
    pprob = 0.8

#q probability value
    qprob = 0.5

# setting Rij values
    rm = np.zeros(SIZE)
    rm[0] = 10

#generates lists for nearests neighbors
    nb = neighgenerate()

#sums of Rij values for random walk
    nsite = np.zeros((NSMOOTH, SIZE))

#number of visits of ite site
    ncounter = np.zeros((NSMOOTH, SIZE), dtype=np.float32)
    nstartcounter = np.zeros((NSMOOTH, SIZE), dtype=np.float32)

    site = np.zeros(NSMOOTH, dtype=np.int32)
    startsite = np.zeros(NSMOOTH, dtype=np.int32)
#starting site
    for p in range(NSMOOTH):
        site[p] = np.random.randint(SIZE)
        startsite[p] = site[p]
        nstartcounter[p, startsite[p]] = 1

    prevprobs = np.zeros(SIZE, dtype=np.float32)

    for _ in range(MAXEPOCHS):

        for p in range(NSMOOTH):

            randsarray = np.random.rand(NSTEPS,3)
            for zz in randsarray:
                z1, z2, z3 = zz
                if (z1 < pprob):
                    ##СМЕРТЬ, рождение на случайном месте
                    site[p] = int(z3 * SIZE)
                    ##Обновление счетчика
                    startsite[p] = site[p]
                    nstartcounter[p, startsite[p]] += 1
                elif (z2 < qprob):
                    ##Переход на соседа
                    site[p] = randomneighbour(nb[site[p]], z3)
                else:
                    ##Переход на благожелательного соседа
                    site[p] = goodneighbor(nb[site[p]], rm, z3)

                ncounter[p,site[p]] += 1
                nsite[p,startsite[p]] += rm[site[p]]

        probs = np.mean(ncounter, axis=0) / NSTEPS
        if (np.max(abs(probs-prevprobs)) < TOLERANCE):
            break

        prevprobs = probs


    #Final values
    result = np.zeros((2,SIZE))
    result[0] = rm
    result[1] = probs

    return

if __name__=='__main__':
    mainsmooth()