import numpy as np

MSIZE = 4
NSIZE = 4
SIZE = MSIZE * NSIZE

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
    NSTEPS = 100000

# p probability value
    pprob = 0.1

#q probability value
    qprob = 0.0

# setting Rij values
    rm = np.zeros(SIZE)
    rm[0] = 10

#generates lists for nearests neighbors
    nb = neighgenerate()
    #print(rm.reshape((-1,MSIZE)))

#sums of Rij values for random walk
    nsite = np.zeros(SIZE)

#number of visits of ite site
    ncounter = np.zeros(SIZE, dtype=int)
    nstartcounter = np.zeros(SIZE, dtype=int)

#starting site
    site = np.random.randint(SIZE)
    startsite = site
    nstartcounter[startsite] = 1

    stepcounter = 1

    for _ in range(2):
 
        probsarray = np.random.rand(NSTEPS,3)

        for zz in probsarray:
            z1, z2, z3 = zz
            if (z1 < pprob):
                site = int(z3 * SIZE)
                startsite = site
                nstartcounter[startsite] += 1
            elif (z2 < qprob):
                neighborsforsite = nb[site]
                site = randomneighbour(neighborsforsite, z3)
            else:
                neighborsforsite = nb[site]
                site = goodneighbor(neighborsforsite, rm, z3)

            ncounter[site] += 1
            nsite[startsite] += rm[site]
            stepcounter += 1

            if (stepcounter % 10000 == 0):
                #print(ncounter.reshape((-1,MSIZE)) / stepcounter)
                print((nsite / nstartcounter).reshape((-1,MSIZE)))


    return

if __name__=='__main__':
    main()