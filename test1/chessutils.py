import numpy as np

def postonote(x):
    '''
    Chess notation
    '''
    numcell = 8
    xp = x % numcell
    yp = x // numcell
    return f"{chr(ord('a')+xp)}{numcell-yp}"

def codetofig(x):
    if x == None:
        return ''
    colordict = {'l':'White', 'd':'Black'}
    figuredict = {'p': 'pawn', 'r': 'rock', 'n': 'knight', 'b': 'bishop', 'q': 'queen', 'k': 'king'}
    return f'{colordict[x[1]]} {figuredict[x[0]]}'

def find_coeffs(pa, pb):
    '''
    Distortion
    '''
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def outres(pred, val):
    if (pred == None):
        print("No chessboard detected")
        return
    numcell = 8
    for i in range(numcell*numcell):
        if (i in pred) and (i in val):
            res = 'Correct' if pred[i] == val[i] else 'False'
            p = codetofig(pred[i])
            v = codetofig(val[i])
            print(f"{i:02d}:({postonote(i)}):\t{res}\tPredicted {p}\tValidation {v}")
        if (i in pred) and not(i in val):
            res = 'False'
            p = codetofig(pred[i])
            print(f"{i:02d}:({postonote(i)}):\t{res}\tPredicted {p}\tValidation empty")
        if not(i in pred) and (i in val):
            res = 'False'
            v = codetofig(val[i])
            print(f"{i:02d}:({postonote(i)}):\t{res}\tPredicted empty\tValidation {v}")
    return

def outpred(pred):
    if (pred == None):
        print("No chessboard detected")
        return
    numcell = 8
    for i in range(numcell*numcell):
        if (i in pred):
            p = codetofig(pred[i])
            print(f"{i:02d}:({postonote(i)}):\tPredicted {p}")
    return
