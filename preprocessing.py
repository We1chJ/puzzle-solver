import os
import cv2

IMAGE_PATH = "test/original.jpeg"     # path to the original puzzle picture
PIECES_PER_ROW = 3            # number of columns (pieces across)
PIECES_PER_COLUMN = 2         # number of rows (pieces down)
OUTPUT_FOLDER = "boxed_images"       # where the box images are saved
SAVE_BOXES = True             # set False to skip writing files


def split_into_grid(image, cols, rows):
    """Split an image into a rows x cols grid.

    Returns a list of (row, col, box_image), row by row from the top left.
    If the size doesn't divide evenly, the last row/column takes the extra pixels.
    """
    height, width = image.shape[:2]
    box_w = width // cols
    box_h = height // rows

    boxes = []
    for r in range(rows):
        for c in range(cols):
            x1 = c * box_w
            y1 = r * box_h
            x2 = width if c == cols - 1 else x1 + box_w
            y2 = height if r == rows - 1 else y1 + box_h
            boxes.append((r, c, image[y1:y2, x1:x2].copy()))
    return boxes


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    boxes = split_into_grid(image, PIECES_PER_ROW, PIECES_PER_COLUMN)

    if SAVE_BOXES:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        for r, c, box in boxes:
            filename = os.path.join(OUTPUT_FOLDER, f"{r}{c}.jpg")
            cv2.imwrite(filename, box)

    print(f"Image size: {image.shape[1]} x {image.shape[0]}")
    print(f"Made {len(boxes)} boxes ({PIECES_PER_COLUMN} rows x {PIECES_PER_ROW} cols)")
    if SAVE_BOXES:
        print(f"Saved to: {OUTPUT_FOLDER}/")

    return boxes


if __name__ == "__main__":
    boxes = main()
