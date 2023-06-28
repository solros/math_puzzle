import imageio.v2 as iio
import numpy as np
import cv2
import random
import sys


def img2tiles(img, rows, cols):
    M = img.shape[0]//rows
    N = img.shape[1]//cols
    tiles = [[img[x:x+M,y:y+N] for x in range(0, M*rows, M)] for y in range(0, N*cols, N)]
    return tiles

def scale_image(img, width=1000):
    orig_width = img.shape[1]
    orig_height = img.shape[0]
    height = int(orig_height/orig_width * width)
    return cv2.resize(img, (width, height))

def tiles2img(tiles):
    n = len(tiles)
    m = len(tiles[0])
    M, N, _ = tiles[0][0].shape
    #print(f"m: {m}, n: {n}, M: {M}, N: {N}")
    img = np.array([[tiles[j][i][I, J] for j in range(n) for J in range(N)] for i in range(m) for I in range(M)])
    return img

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


def add_border(img, col=np.array([0,0,0]), width=1):
    for i in range(img.shape[0]):
        for w in range(width):
            img[i][w] = col
            img[i][img.shape[1]-1-w] = col
    for j in range(img.shape[1]):
        for w in range(width):
            img[w][j] = col
            img[img.shape[0]-1-w][j] = col

def add_text(img, txt, size=2, color=(255,255,255), bordercol=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = size
    thickness = size
    textsize = cv2.getTextSize(txt, font, fontScale, 2*thickness)[0]
    org = ((img.shape[1]-textsize[0])//2, (img.shape[0]+textsize[1])//2)
    r = cv2.putText(img, txt, org, font, fontScale, bordercol, 2*thickness, cv2.LINE_AA)
    return cv2.putText(r, txt, org, font, fontScale, color, thickness, cv2.LINE_AA)


def random_exercise_times(maxval=10, minval=0):
    a = random.randint(minval, maxval)
    b = random.randint(minval, maxval)
    c = a * b
    return f"{a}*{b}", str(c)

def random_exercise_sum(maxval=10, minval=0):
    a = random.randint(minval, maxval)
    b = random.randint(minval, maxval)
    c = a + b
    return f"{a}+{b}", str(c)

def random_exercises(f, n):
    ex_list = []
    ans_list = []
    while len(ex_list) < n:
        q, a = f()
        if a not in ans_list:
            ex_list.append((q, a))
            ans_list.append(a)
    return ex_list

def tile_and_patch(fn, m, n):
    img = iio.imread(fn)
    img2 = tiles2img(img2tiles(img, m, n))
    iio.imwrite(f"{fn.split('.')[0]}_2.jpg", img2)

def tile_permute_patch(fn, m, n):
    img = iio.imread(fn)
    img = scale_image(img)
    tiles = img2tiles(img, m, n)
    grid = create_empty_grid(tiles)
    ex_list = random_exercises(lambda: random_exercise_times(10, 0), m*n)
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
    iio.imwrite(f"{fn.split('.')[0]}_out.jpg", both)
    

if __name__ == "__main__":
    tile_permute_patch(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))