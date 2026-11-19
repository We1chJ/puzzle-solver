import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def display_images(original_image, original_boxes, pieces):

    original_image = mpimg.imread(original_image)

    # Load reference boxes
    list_of_boxes = []
    box_coordinates = []

    for box in original_boxes:
        read_box = mpimg.imread(box)
        list_of_boxes.append(read_box)

    # Load puzzle pieces
    list_of_pieces = []

    for piece in pieces:
        read_piece = mpimg.imread(piece)
        list_of_pieces.append(read_piece)

    # Calculate box coordinates from the original image dimensions
    height, width = original_image.shape[:2]

    cols = 3
    rows = 2

    box_w = width // cols
    box_h = height // rows

    for r in range(rows):
        for c in range(cols):

            x1 = c * box_w
            y1 = r * box_h

            x2 = width if c == cols - 1 else x1 + box_w
            y2 = height if r == rows - 1 else y1 + box_h

            box_coordinates.append((x1, y1, x2 - x1, y2 - y1))

    # Display images
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))

    # Display original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")

    for i, box in enumerate(list_of_boxes):
        axes[1, i].imshow(box)
        axes[1, i].set_title(f"Box {i}")

    for i, piece in enumerate(list_of_pieces):
        axes[2, i].imshow(piece)
        axes[2, i].set_title(f"Piece {i+1}")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return original_image, list_of_boxes, list_of_pieces, box_coordinates


def template_matching(original_image, original_boxes, pieces, box_coordinates):
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
        box_coordinates: The coordinates of the boxes in the original image.

    Returns:
        solved_puzzle: The rearranged puzzle pieces that match the original image.
        matches: The match score for each piece and the corresponding box it was matched to.
    """

    solved_puzzle = np.zeros_like(original_image)

    # Piece-to-box assignments
    matches = {}

    # Identifies boxes that have or have not been filled
    available_boxes = list(range(len(original_boxes)))

    # Convert image to uint8
    def convert_image(image):
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = image * 255
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    for piece_id, piece in enumerate(pieces):

        best_score = -1
        best_box = None
        best_piece = None

        # Convert puzzle piece
        piece = convert_image(piece)

        # Convert to grayscale
        if piece.ndim == 3:
            piece_gray = cv2.cvtColor(piece, cv2.COLOR_RGB2GRAY)
        else:
            piece_gray = piece

        for box_id in available_boxes:

            box = convert_image(original_boxes[box_id])

            h, w = box.shape[:2]

            # Resize puzzle piece
            piece_resized = cv2.resize(piece, (w, h))

            # Convert reference box to grayscale
            if box.ndim == 3:
                box_gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
            else:
                box_gray = box

            # Convert resized piece to grayscale
            if piece_resized.ndim == 3:
                piece_gray_resized = cv2.cvtColor(piece_resized, cv2.COLOR_RGB2GRAY)
            else:
                piece_gray_resized = piece_resized

            # Ensure matching data types
            box_gray = box_gray.astype(np.uint8)
            piece_gray_resized = piece_gray_resized.astype(np.uint8)

            # Compare reference box with puzzle piece
            result = cv2.matchTemplate(
                box_gray, piece_gray_resized, cv2.TM_CCOEFF_NORMED
            )

            score = float(result[0][0])

            if score > best_score:
                best_score = score
                best_box = box_id
                best_piece = piece_resized.copy()

        # Assign piece to best box
        if best_box is not None:

            x, y, w, h = box_coordinates[best_box]

            best_piece = cv2.resize(best_piece, (w, h))

            # Convert piece to match original image channels
            if best_piece.ndim == 2 and solved_puzzle.ndim == 3:
                best_piece = cv2.cvtColor(best_piece, cv2.COLOR_GRAY2RGB)

            if best_piece.ndim == 3 and solved_puzzle.ndim == 2:
                best_piece = cv2.cvtColor(best_piece, cv2.COLOR_RGB2GRAY)

            # Convert to original image dtype
            best_piece = best_piece.astype(solved_puzzle.dtype)

            solved_puzzle[y : y + h, x : x + w] = best_piece

            # Match score
            matches[piece_id] = {
                "box": best_box,
                "score": best_score,
                "location": (x, y),
            }

            print(
                f"Piece {piece_id + 1} -> Box {best_box}: " f"Score = {best_score:.4f}"
            )

            available_boxes.remove(best_box)

    return solved_puzzle, matches
