import cv2


def apply_binary(img, bin_threshold):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if bin_threshold == 0 or bin_threshold == 255:
        # Otsu's threshold
        otsu_threshold, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print("Otsu threshold:", otsu_threshold)
    else:
        _, img_bin = cv2.threshold(img_gray, bin_threshold, 255, cv2.THRESH_BINARY_INV)

    return cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
