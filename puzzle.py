import imageio.v2 as iio
import numpy as np
import cv2
import random
import sys
import argparse


desc = """This script creates math (or vocabulary, etc.) puzzles from images.
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('file', type=str, help="path to input image file")
parser.add_argument('rows', type=int, help="number of puzzle rows")
parser.add_argument('cols', type=int, help="number of puzzle columns")
parser.add_argument('-o', '--outfile', type=str, help="output file name")
parser.add_argument('--noscale', action="store_true", help="don't scale image, keep original size")
parser.add_argument('--width', type=int, default=1000, help="width of final image")
parser.add_argument('--height', type=int, default=0, help="height of final image")
parser.add_argument('--multi', action="store_true", help="use multiplication exercises")
parser.add_argument('--multimaxfactor', type=int, default=10, help="maximal factor in multiplication exercises")
parser.add_argument('--multiminfactor', type=int, default=0, help="minimal factor in multiplication exercises")
parser.add_argument('--multimaxresult', type=int, default=100, help="maximal result in multiplication exercises")
parser.add_argument('--multiminresult', type=int, default=0, help="minimal result in multiplication exercises")
parser.add_argument('--plus', action="store_true", help="use addition exercises")
parser.add_argument('--plusmaxsummand', type=int, default=10, help="maximal factor in addition exercises")
parser.add_argument('--plusminsummand', type=int, default=0, help="minimal factor in addition exercises")
parser.add_argument('--plusmaxresult', type=int, default=20, help="maximal result in addition exercises")
parser.add_argument('--plusminresult', type=int, default=0, help="minimal result in addition exercises")
args = parser.parse_args()

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
    img = np.array([[tiles[j][i][I, J] for j in range(n) for J in range(N)] for i in range(m) for I in range(M)])
    return img

# scale image
def scale_image(img, width=1000, height=0):
    orig_width = img.shape[1]
    orig_height = img.shape[0]
    if height == 0:
        height = int(orig_height/orig_width * width)
    return cv2.resize(img, (width, height))

# apply random permutation to tiles in mxn array
def permute_tiles(tiles):
    n = len(tiles)
    m = len(tiles[0])
    #print(f"m: {m}, n: {n}")
    perm = np.random.permutation(range(m*n))
    #print(f"perm: {perm}")
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
    if not (args.multi or args.plus):
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
    while len(ex_list) < n:
        q, a = random_exercise(ex_types)
        if a not in ans_list:
            ex_list.append((q, a))
            ans_list.append(a)
        else:
            fails += 1
            if fails >= n**2:
                raise Exception("cannot find enough unique exercises")
    return ex_list

def exercise_types(args):
    ex_types = []
    if args.multi:
        ex_types.append(MultiExercise(args.multimaxfactor, args.multiminfactor, args.multimaxresult, args.multiminresult))
    if args.plus:
        ex_types.append(PlusExercise(args.plusmaxsummand, args.plusminsummand, args.plusmaxresult, args.plusminresult))
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

class PlusExercise(Exercise):
    def qa(self):
        if self.maxval > self.maxres:
            self.maxval = self.maxres
        a = self.get_random_operand()
        minval = self.minval if a >= self.minres else self.minres - a
        maxval = self.maxval if self.maxres - a >= self.maxval else self.maxres - a
        b = self.get_random_operand(minval, maxval)

        c = a + b
        if c < self.minres or c > self.maxres:
            raise Exception("result error in plus")

        return f"{a}+{b}", str(c)

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


def tile_and_patch(fn, m, n):
    img = iio.imread(fn)
    img2 = tiles2img(img2tiles(img, m, n))
    iio.imwrite(f"{fn.split('.')[0]}_2.jpg", img2)

def tile_permute_patch(args):
    fn = args.file
    m = args.rows
    n = args.cols
    img = iio.imread(fn)
    img = scale_image(img)
    tiles = img2tiles(img, m, n)
    grid = create_empty_grid(tiles)
    #ex_list = random_exercises(lambda: random_exercise_times(10, 0), m*n)
    ex_list = random_exercises(args)
    print(ex_list)
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
    #iio.imwrite(f"{fn.split('.')[0]}_2.jpg", img2)
    #iio.imwrite(f"{fn.split('.')[0]}_grid.jpg", grid2)
    both = tiles2img([[grid2, img2]])
    outfile = args.outfile if args.outfile else f"{fn.split('.')[0]}_out.jpg"
    iio.imwrite(outfile, both)
    

if __name__ == "__main__":
    tile_permute_patch(args)