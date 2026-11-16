from __future__ import print_function
import glob
import os
import sys

import cv2 as cv
import numpy as np

## [global_variables]
image_window = "Source Image"
solved_window = "Solved Puzzle"
SOLVED_PATH = "solved.png"

# The pieces are photographed at a different scale than the original image, so
# each piece is resized over this range of factors before template matching.
MIN_SCALE = 0.3
MAX_SCALE = 1.2
SCALE_STEP = 0.02

# A piece is accepted as soon as it matches the original at least this well,
# so the remaining scales don't have to be searched
ACCEPT_THRESHOLD = 0.8

# Pixels this dark in a cropped piece are background left over from masking
BACKGROUND_THRESHOLD = 10
## [global_variables]


def main(argv):

    if len(argv) < 2:
        print('Not enough parameters')
        print('Usage:\ntesting_matching.py <original_image> <pieces_dir> [<rows> <cols>]')
        return -1

    ## [load_image]
    img = cv.imread(argv[0], cv.IMREAD_COLOR)
    piece_paths = sorted(glob.glob(os.path.join(argv[1], "piece_*.png")))
    rows, cols = (int(argv[2]), int(argv[3])) if len(argv) > 3 else (2, 3)

    if img is None or not piece_paths:
        print('Can\'t read the original image or find any pieces')
        return -1
    ## [load_image]

    placements, solved = Solve(img, piece_paths, rows, cols)

    for path, (row, col) in placements.items():
        print('{}: row {}, col {}'.format(os.path.basename(path), row, col))

    cv.imwrite(SOLVED_PATH, solved)
    print('Solved puzzle saved to', SOLVED_PATH)

    ## [imshow]
    img_display = img.copy()
    cell_h, cell_w = img.shape[0] / rows, img.shape[1] / cols
    for path, (row, col) in placements.items():
        cv.rectangle(img_display, (int(col * cell_w), int(row * cell_h)),
                     (int((col + 1) * cell_w), int((row + 1) * cell_h)), (0, 255, 0), 2)
        cv.putText(img_display, os.path.splitext(os.path.basename(path))[0],
                   (int((col + 0.5) * cell_w) - 30, int((row + 0.5) * cell_h)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.namedWindow(image_window, cv.WINDOW_AUTOSIZE)
    cv.namedWindow(solved_window, cv.WINDOW_AUTOSIZE)
    cv.imshow(image_window, img_display)
    cv.imshow(solved_window, solved)
    ## [imshow]

    ## [wait_key]
    cv.waitKey(0)
    return 0
    ## [wait_key]


def Solve(img, piece_paths, rows, cols):
    """
    Find the grid cell of every piece and assemble the solved puzzle.

    Returns:
        placements: {piece path: (row, col)}
        solved: the pieces pasted where they belong, the same size as img
    """
    cell_h, cell_w = img.shape[0] / rows, img.shape[1] / cols
    solved = np.zeros_like(img)

    matches = []
    for path in piece_paths:
        templ = cv.imread(path, cv.IMREAD_COLOR)
        score, loc, resized, mask = MatchingMethod(img, templ)
        if score < ACCEPT_THRESHOLD:
            print('Warning: {} only matched {:.3f}, using its best match'.format(
                os.path.basename(path), score))

        matches.append(CellOverlap(mask, loc, rows, cols, cell_h, cell_w))

        # Paste only the piece's own pixels, so neighbouring tabs and holes interlock
        h, w = mask.shape
        region = solved[loc[1]:loc[1] + h, loc[0]:loc[0] + w]
        region[mask > 0] = resized[mask > 0]

    # Give each cell to one piece, taking the most confident matches first
    cells = AssignCells(matches, rows, cols)
    placements = {path: cells[i] for i, path in enumerate(piece_paths)}
    return placements, solved


def MatchingMethod(img, templ):
    """
    Find where a piece sits in the original image.

    The piece is resized over a range of scales, and at each scale it is
    matched against the original using only its own pixels (the black
    background around the piece and inside its holes is masked out). The
    search stops at the first scale that passes ACCEPT_THRESHOLD, then steps
    up or down a few scales to settle on the best fit, so the piece can be
    pasted at the right size.

    Returns:
        (score, top-left location, resized piece, piece mask)
    """
    best = None
    best_scale = MIN_SCALE

    for scale in np.arange(MIN_SCALE, MAX_SCALE + 1e-9, SCALE_STEP):
        scale = round(float(scale), 2)
        match = MatchAtScale(img, templ, scale)
        # The piece has to fit inside the original image
        if match is None:
            break

        if best is None or match[0] > best[0]:
            best, best_scale = match, scale
        if best[0] >= ACCEPT_THRESHOLD:
            break

    if best is None:
        raise ValueError('Piece is too large to fit in the original image at any scale')

    # Neighbouring scales usually fit better than the first one accepted
    for step in (SCALE_STEP, -SCALE_STEP):
        while True:
            scale = round(best_scale + step, 2)
            match = MatchAtScale(img, templ, scale)
            if match is None or match[0] <= best[0]:
                break
            best, best_scale = match, scale

    return best


def MatchAtScale(img, templ, scale):
    """
    Match a piece against the original at a single scale.

    Returns:
        (score, top-left location, resized piece, piece mask), or None if
        the resized piece doesn't fit inside the original
    """
    resized = cv.resize(templ, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    h, w = resized.shape[:2]
    if h > img.shape[0] or w > img.shape[1]:
        return None

    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    mask = (gray > BACKGROUND_THRESHOLD).astype(np.uint8) * 255

    ## [match_template]
    result = cv.matchTemplate(img, resized, cv.TM_CCOEFF_NORMED, None, mask)
    result[~np.isfinite(result)] = -1
    ## [match_template]

    ## [best_match]
    _minVal, maxVal, _minLoc, maxLoc = cv.minMaxLoc(result)
    ## [best_match]

    return maxVal, maxLoc, resized, mask


def CellOverlap(mask, loc, rows, cols, cell_h, cell_w):
    """
    How much of the matched piece falls in each grid cell. The tabs of a piece
    stick out into its neighbours, so the cell with the most overlap is the one
    the piece's body sits in.
    """
    full_mask = np.zeros((int(round(rows * cell_h)), int(round(cols * cell_w))), np.uint8)
    h, w = mask.shape
    x, y = loc
    full_mask[y:y + h, x:x + w] = mask

    overlap = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            cell = full_mask[int(r * cell_h):int((r + 1) * cell_h), int(c * cell_w):int((c + 1) * cell_w)]
            overlap[r, c] = np.count_nonzero(cell) / cell.size
    return overlap


def AssignCells(cell_overlaps, rows, cols):
    """
    Greedily give each cell to the piece that overlaps it most, so that no two
    pieces end up in the same place.
    """
    candidates = []
    for i, overlap in enumerate(cell_overlaps):
        for r in range(rows):
            for c in range(cols):
                candidates.append((overlap[r, c], i, (r, c)))
    candidates.sort(reverse=True)

    placements = {}
    taken = set()
    for _overlap, i, cell in candidates:
        if i in placements or cell in taken:
            continue
        placements[i] = cell
        taken.add(cell)
    return placements


if __name__ == "__main__":
    main(sys.argv[1:])
