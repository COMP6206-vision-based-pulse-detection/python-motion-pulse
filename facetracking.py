
def make_face_rects(rect):
    """ Given a rectangle (covering a face), return two rectangles.
        which cover the forehead and the area under the eyes """
    x, y, w, h = rect

    rect1_x = x + w / 4.0
    rect1_w = w / 2.0
    rect1_y = y + 0.05 * h
    rect1_h = h * 0.9 * 0.2

    rect2_x = rect1_x
    rect2_w = rect1_w

    rect2_y = y + 0.05 * h + (h * 0.9 * 0.55)
    rect2_h = h * 0.9 * 0.45

    return (
        (int(rect1_x), int(rect1_y), int(rect1_w), int(rect1_h)),
        (int(rect2_x), int(rect2_y), int(rect2_w), int(rect2_h))
    )