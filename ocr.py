import pytesseract
import numpy as np

def read(img):
    img = img - img.min()
    img = img / img.max()
    img = img * 255
    img = img.astype(np.uint8)
    img = pytesseract.image_to_string(img)

    return img
