import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./real.png")
# cv2.imshow("Original Image", img)

# color masking
bg_color = np.mean(img[0:100, 0:100], axis=(0, 1))
# we assume that all pieces will be within the view of the photo and not partially covered by the border, thus clear to use the corner patch
# show how big the box is that is used to calculate the background color
# cv2.rectangle(img, (0, 0), (100, 100), (0, 255, 0), 2)
# cv2.imshow("Original Image", img)

# print("background colour (BGR):", bg_color)

dist = np.linalg.norm(img - bg_color, axis=2)

# far from background = piece
T = 25
piece_mask = (dist > T).astype(np.uint8) * 255
cv2.imshow("Piece mask", piece_mask)

# close: fill gaps on the pieces that are considered missing
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("Closed mask", closed)

# open: remove small specks in the background
small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, small_kernel)
cv2.imshow("Clean mask", clean_mask)

# find contours
contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# remove any small blobs found
AREA_THRESHOLD = 40000
pieces = [c for c in contours if cv2.contourArea(c) > AREA_THRESHOLD]
print("pieces found:", len(pieces))


# draw the outlines on a copy of the original
outlined = img.copy()
cv2.drawContours(outlined, pieces, -1, (0, 0, 255), 2)
cv2.imshow("Filtered outlines", outlined)

# mask containing only the kept pieces (filled, specks removed)
filtered_mask = np.zeros_like(clean_mask)
cv2.drawContours(filtered_mask, pieces, -1, 255, -1)
cv2.imshow("Filtered mask", filtered_mask)

# distribution of contour areas (log scale: specks are tiny, pieces are huge)
areas = [cv2.contourArea(c) for c in contours]
print("kept areas (sorted):", sorted(int(a) for a in areas if a > AREA_THRESHOLD))
plt.hist(areas, bins=np.logspace(0, np.log10(max(areas) + 1), 50))
plt.xscale("log")
plt.axvline(AREA_THRESHOLD, color="red", label="AREA_THRESHOLD")
plt.xlabel("contour area (pixels)")
plt.ylabel("count")
plt.legend()
plt.show()

# draw bounding boxes, and crop it out on the masked image
masked_img = cv2.bitwise_and(img, img, mask=clean_mask)
for i, contour in enumerate(pieces):
    x, y, w, h = cv2.boundingRect(contour)
    piece_crop = masked_img[y:y + h, x:x + w]
    # cv2.imshow(f"Piece {i + 1}", piece_crop)
    # save each cropped piece to a file
    cv2.imwrite(f"./test/piece_{i + 1}.png", piece_crop)


# wait key destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()