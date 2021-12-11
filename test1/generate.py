import numpy as np
import cv2
import os
import uuid
from PIL import Image
from boardgen import moirebackground
from boardgen import chessboard
import argparse

# NUMCELL = 8
# CELL = 45
# BOARDSIZE = CELL * NUMCELL
IMGSIZE = 480
MAXSHEAR = 0.05
MINSCALE = 0.5

figs = ['p', 'b', 'n', 'r', 'q', 'k']
colors = ['d', 'l']


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

    parameters = {
        'numcell':8,
        'cellsize':45,
        'figures':figs,
        'colors':colors,
        'shear':MAXSHEAR,
        'scale':MINSCALE,
    }

    for _ in range(args.number):

        #for a filename
        idx = uuid.uuid4() 

        #random board with figure positions
        boardimage, recs, _ = chessboard(figuresimgs, np.random.rand(13), IMGSIZE, parameters)

        if (args.moire):
            #random moire pattern
            moir = moirebackground(np.random.rand(8), IMGSIZE)
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