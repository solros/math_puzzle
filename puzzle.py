import numpy as np
import cv2
import random
import logging
import argparse
#from matplotlib import pyplot as plt


desc = """This script creates math (or vocabulary, etc.) puzzles from images.
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('file', type=str, help="path to input image file")
parser.add_argument('rows', type=int, help="number of puzzle rows")
parser.add_argument('cols', type=int, help="number of puzzle columns")
parser.add_argument('-o', '--outfile', type=str, help="output file name")
parser.add_argument('-s', '--split', action="store_true", help="split into two files")
parser.add_argument('--noscale', action="store_true", help="don't scale image, keep original size")
parser.add_argument('--width', type=int, default=1000, help="width of final image")
parser.add_argument('--height', type=int, default=0, help="height of final image")
parser.add_argument('--hdist', type=int, default=0, help="distance between grid and puzzle")
parser.add_argument('--rotate', action="store_true", help="rotate image and place side by side (instead of on top of each other)")
parser.add_argument('--multi', action="store_true", help="use multiplication exercises")
parser.add_argument('--multimaxfactor', type=int, default=10, help="maximal factor in multiplication exercises")
parser.add_argument('--multiminfactor', type=int, default=0, help="minimal factor in multiplication exercises")
parser.add_argument('--multimaxresult', type=int, default=100, help="maximal result in multiplication exercises")
parser.add_argument('--multiminresult', type=int, default=0, help="minimal result in multiplication exercises")
parser.add_argument('--plus', action="store_true", help="use plus exercises")
parser.add_argument('--minus', action="store_true", help="use minus exercises")
parser.add_argument('--plusmaxsummand', type=int, default=10, help="maximal summand in addition exercises")
parser.add_argument('--plusminsummand', type=int, default=0, help="minimal summand in addition exercises")
parser.add_argument('--plusmaxresult', type=int, default=20, help="maximal result in addition exercises")
parser.add_argument('--plusminresult', type=int, default=0, help="minimal result in addition exercises")
parser.add_argument('--custom', type=str, help="file (xslx or csv) containing custom exercises")
parser.add_argument('--custom_questions', type=int, default=0, help="what column should be used as questions; 0 (default): both at random, 1: first col, 2: second col")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    filename='puzzle.log',
                    encoding='utf-8', 
                    level=logging.DEBUG)

logger = logging.getLogger('puzzle_generator')

# split image into tiles
def img2tiles(img, rows, cols):
    M = img.shape[0]//rows
    N = img.shape[1]//cols
    tiles = [[img[x:x+M,y:y+N] for x in range(0, M*rows, M)] for y in range(0, N*cols, N)]
    return tiles

# merge tiles into image
def tiles2img(tiles):
    n = len(tiles)
    m = len(tiles[0])
    M, N, _ = tiles[0][0].shape
    img = np.array([[tiles[j][i][I, J] for j in range(n) for J in range(N)] for i in range(m) for I in range(tiles[0][i].shape[0])])
    return img

# scale image
def scale_image(img, width=1000, height=0):
    orig_width = img.shape[1]
    orig_height = img.shape[0]
    if height == 0:
        height = int(orig_height/orig_width * width)
    logger.debug(f"Scaling to {width}x{height}")
    return cv2.resize(img, (width, height))

# apply random permutation to tiles in mxn array
def permute_tiles(tiles):
    n = len(tiles)
    m = len(tiles[0])
    #print(f"m: {m}, n: {n}")
    perm = np.random.permutation(range(m*n))
    logger.debug(f"Using permutation {perm}")
    ptiles = np.copy(tiles)
    for i in range(m):
        for j in range(n):
            #print(f"perm: i: {i}, j: {j}, i*n+j={i*n+j}")
            p = perm[i * n + j]
            #print(f"p = {p}, coords: {p//m}, {p%n}")
            ptiles[j][i] = tiles[p//m][p%m] 
    return ptiles


def create_empty_grid(tiles):
    grid = [[np.full((tiles[0][0]).shape, 255, dtype=np.uint8) for _ in row] for row in tiles]
    for r in grid:
        for c in r:
            add_border(c)
    return grid

# add border to an image
def add_border(img, col=np.array([0,0,0]), width=1):
    for i in range(img.shape[0]):
        for w in range(width):
            img[i][w] = col
            img[i][img.shape[1]-1-w] = col
    for j in range(img.shape[1]):
        for w in range(width):
            img[w][j] = col
            img[img.shape[0]-1-w][j] = col


#def add_text_plt(img)

# add text to the center of an image
# default: white with black border
def add_text(img, txt, size=2, color=(255,255,255), bordercol=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = size
    thickness = size
    textsize = cv2.getTextSize(txt, font, fontScale, 2*thickness)[0]
    org = ((img.shape[1]-textsize[0])//2, (img.shape[0]+textsize[1])//2)
    r = cv2.putText(img, txt, org, font, fontScale, bordercol, 2*thickness, cv2.LINE_AA)
    return cv2.putText(r, txt, org, font, fontScale, color, thickness, cv2.LINE_AA)



def random_exercises(args):
    if not (args.multi or args.plus or args.minus or args.custom):
        args.multi = True
        #raise Exception("No available exercise types")

    ex_types = exercise_types(args)

    # number of exercises needed
    n = args.rows * args.cols
    # list of exercises as (question, answer) tuples
    ex_list = []
    # list of answers (to avoid duplicates)
    ans_list = []

    fails = 0
    logger.debug(f"Trying to create {n} exercises")
    while len(ex_list) < n:
        q, a = random_exercise(ex_types)
        logger.debug(f"{q}={a}")
        if a not in ans_list:
            ex_list.append((q, a))
            ans_list.append(a)
        else:
            fails += 1
            logger.debug("Solution already exists")
            if fails >= n**2:
                logger.error(f"Cannot find enough unique exercises after {fails} fails.")
                raise Exception("cannot find enough unique exercises")
    logger.info(f"exercises: {ex_list}")
    return ex_list

def exercise_types(args):
    ex_types = []
    if args.multi:
        logger.debug("Using multi")
        ex_types.append(MultiExercise(args.multimaxfactor, args.multiminfactor, args.multimaxresult, args.multiminresult))
    if args.plus:
        logger.debug("Using plus")
        ex_types.append(PlusExercise(args.plusmaxsummand, args.plusminsummand, args.plusmaxresult, args.plusminresult))
    if args.minus:
        logger.debug("Using minus")
        ex_types.append(MinusExercise(args.plusmaxsummand, args.plusminsummand, args.plusmaxresult, args.plusminresult))
    if args.custom:
        logger.debug("Using custom")
        ex_types.append(CustomExercise(args.custom, args.custom_questions))
    return ex_types

def random_exercise(ex_types):
    e = random.choice(ex_types)
    return e.qa()

class Exercise:
    def __init__(self, maxval=10, minval=0, maxres=1000, minres=0):
        self.maxval = maxval
        self.minval = minval
        self.maxres = maxres
        self.minres = minres

        if minval > maxval or minres > maxres:
            raise Exception("incompatible min/max values")
    
    def qa(self):
        pass

    def get_random_operand(self, minval=None, maxval=None):
        if minval is None:
            minval = self.minval
        if maxval is None:
            maxval = self.maxval
        return random.randint(minval, maxval)

class PlusMinusExercise(Exercise):
    def ops(self):
        if self.maxval > self.maxres - self.minval:
            self.maxval = self.maxres - self.minval
        a = self.get_random_operand()
        minval = self.minval if a >= self.minres else self.minres - a
        maxval = self.maxval if self.maxres - a >= self.maxval else self.maxres - a
        b = self.get_random_operand(minval, maxval)

        c = a + b
        if c < self.minres or c > self.maxres:
            raise Exception("result error in plus")

        return a, b, c

class PlusExercise(PlusMinusExercise):
    def qa(self):
        a, b, c = self.ops()
        return f"{a}+{b}", str(c)

class MinusExercise(PlusMinusExercise):
    def qa(self):
        a, b, c = self.ops()
        return f"{c}-{a}", str(b)


class MultiExercise(Exercise):
    def qa(self):
        if self.maxval > self.maxres:
            self.maxval = self.maxres
        
        a = self.get_random_operand()
        minval = self.minval if a >= self.minres else self.minres//a + 1
        maxval = self.maxval if a == 0 or self.maxres // a >= self.maxval else self.maxres // a
        b = self.get_random_operand(minval, maxval)

        c = a * b
        if c < self.minres or c > self.maxres:
            raise Exception("result error in times")

        return f"{a}*{b}", str(c)

class RootExercise(Exercise):
    def qa(self):
        import math
        if self.maxval > self.maxres**2:
            self.maxres = int(math.sqrt(self.maxval))
        
        c = self.get_random_operand(self.minres, self.maxres)
        a = c**2

        return f"sqrt({a})", str(c)

class CustomExercise(Exercise):
    def __init__(self, fn, questions=0):
        self.questions = questions
        import pandas as pd
        logging.info(f"Reading custom exercises from {fn}")
        extension = fn.split(".")[-1]
        if extension == "xslx":
            self.df = pd.read_excel(fn, header=None)
        elif extension == "csv":
            self.df = pd.read_csv(fn, header=None)
        print(self.df)
        self.index = self.df.index.to_list()
    
    def qa(self):
        ind = random.choice(self.index)
        q, a = self.df[0][ind], self.df[1][ind]
        logger.debug(f"Randomly picked custom exercise {q}: {a} at index {ind}")
        if self.questions == 0:
            if random.random() >= 0.5:
                logger.debug("Swapping order")
                q, a = a, q
        elif self.questions == 2:
            q, a = a, q
        return q, a

        


def tile_and_patch(fn, m, n):
    img = cv2.imread(fn)
    img2 = tiles2img(img2tiles(img, m, n))
    cv2.imwrite(f"{fn.split('.')[0]}_2.jpg", img2)

def tile_permute_patch(args):
    fn = args.file
    m = args.rows
    n = args.cols
    logger.info(f"Making {fn} into {m}x{n} puzzle")
    img = cv2.imread(fn)
    if not args.noscale:
        img = scale_image(img, args.width, args.height)
    tiles = img2tiles(img, m, n)
    grid = create_empty_grid(tiles)
    ex_list = random_exercises(args)
    print(list(map(lambda a: f"{a[0]}={a[1]}", ex_list)))
    i = -1
    for row in range(len(tiles)):
        for col in range(len(tiles[row])):
            i += 1
            q, a = ex_list[i]
            t = tiles[row][col]
            add_border(t)
            t = add_text(t, a)
            grid[row][col] = add_text(grid[row][col], q, size=2, color=(0,0,0))
            tiles[row][col] = t
    tiles = permute_tiles(tiles)
    img2 = tiles2img(tiles)
    grid2 = tiles2img(grid)
    if args.rotate:
        img2 = np.rot90(img2, 3)
        grid2 = np.rot90(grid2, 3)

    outfile = args.outfile if args.outfile else f"{fn.split('.')[0]}_out.jpg"
    if args.split:
        cv2.imwrite(outfile, img2)
        outfile_grid = f"{'.'.join(outfile.split('.')[:-1])}_grid.{outfile.split('.')[-1]}"
        cv2.imwrite(outfile_grid, grid2)
    else:
        dist = np.full((args.hdist, img2.shape[1], 3), 255, dtype=np.uint8)
        both = tiles2img([[grid2, dist, img2]])
        logger.info(f"Writing output to file {outfile}")
        cv2.imwrite(outfile, both)
    

import matplotlib.pyplot as plt
import io
from PIL import Image, ImageChops

white = (255, 255, 255, 255)
def latex_to_img(tex):
    buf = io.BytesIO()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.axis('off')
    plt.text(0.05, 0.5, f'${tex}$', size=40)
    plt.savefig(buf, format='png')
    plt.close()
#
    im = Image.open(buf)
    bg = Image.new(im.mode, im.size, white)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return im.crop(bbox)


if __name__ == "__main__":
    tile_permute_patch(args)