

def calc_diff(img1, img2):
    diff = img1 - img2
    diff = abs(diff)
    return diff.max()
