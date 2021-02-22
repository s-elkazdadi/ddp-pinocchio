from PIL import Image
import sys
import os


def rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc - minc) / maxc
    rc = (maxc - r) / (maxc - minc)
    gc = (maxc - g) / (maxc - minc)
    bc = (maxc - b) / (maxc - minc)
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h / 6.0) % 1.0
    return h, s, v


GREEN_RANGE_MIN_HSV = (100, 80, 70)
GREEN_RANGE_MAX_HSV = (185, 255, 255)


def main():
    for file_path in sys.argv[1:]:
        name, ext = os.path.splitext(file_path)
        print(name)
        im = Image.open(file_path)
        im = im.convert("RGBA")

        # Go through all pixels and turn each 'green' pixel to transparent
        pix = im.load()
        width, height = im.size
        for x in range(width):
            for y in range(height):
                r, g, b, a = pix[x, y]
                h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

                min_h, min_s, min_v = GREEN_RANGE_MIN_HSV
                max_h, max_s, max_v = GREEN_RANGE_MAX_HSV
                if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
                    pix[x, y] = (0, 0, 0, 0)

        im.save("alpha_" + name + ".png")


if __name__ == "__main__":
    main()
