import cv2
import numpy as np
from matplotlib import pyplot as plt


def template_matching(original_image, original_boxes, pieces):
    """
    Template matching function to fit puzzle pieces into the original
        image. It loops through each puzzle piece, compares it against each
        box of the original image, and calculates the percentage match. The
        piece is assigned to the box with the highest similarity and placed
        in the corresponding position.

    Args:
        original_image: The original image into which pieces need to be fit.
        original_boxes: The boxed segments of the original image.
        pieces: A list of puzzle pieces to be matched.

    Returns:
        solved_puzzle: The rearranged puzzle pieces that match the original image.
    """
    # Blank image template to place puzzle pieces
    solved_puzzle = np.zeros_like(original_image)

    # Piece-to-box assignments
    matches = {}

    # Identifies boxes that have or have not been filled
    available_boxes = list(range(len(original_boxes)))

    # Loop through each puzzle piece
    for piece_id, piece in enumerate(pieces):

        best_score = -1
        best_box = None

        # Compare piece with every original box
        for box_id in available_boxes:

            # Extract box coordinates
            x, y, w, h = original_boxes[box_id]

            # Extract box from original image
            box = original_image[y : y + h, x : x + w]

            # Resize piece to match the box size
            piece_resized = cv2.resize(piece, (w, h))

            # OpenCV template matching
            result = cv2.matchTemplate(box, piece_resized, cv2.TM_CCOEFF_NORMED)

            # Get maximum correlation score
            max_val = cv2.minMaxLoc(result)[1]

            # Check if this is the best match
            if max_val > best_score:
                best_score = max_val
                best_box = box_id

        # Assign piece to best box
        if best_box is not None:

            x, y, w, h = original_boxes[best_box]

            # Place piece in blank template
            solved_puzzle[y : y + h, x : x + w] = piece_resized

            matches[piece_id] = {"box": best_box, "score": best_score}

            available_boxes.remove(best_box)

    return solved_puzzle, matches
