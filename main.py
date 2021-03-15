import cv2 as cv2
import numpy as np
from mss import mss
import matplotlib.pyplot as plt

captured_monitor = 2
image_height_bottom = 640
image_height_top = 375


def make_points(line):
    slope, intercept = line
    y1 = int(image_height_bottom)
    y2 = int(y1 * 5 / 7)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        fit = np.polyfit((x1, x2), (y1, y2), 1)
        slope = fit[0]
        intercept = fit[1]
        if slope < 0:  # y is reversed in image
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    try:
        left_line = make_points(left_fit_average)
        right_line = make_points(right_fit_average)
        return [left_line, right_line]
    except Exception as e:
        print(e, '\n')
        return None


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150)


def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if is_safe_integer(x1) and is_safe_integer(x2) and is_safe_integer(y1) and is_safe_integer(y2):
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def is_safe_integer(test_int):
    # temporary workaround for internal library error when converting int
    return -2147483648 < test_int < 2147483647


def region_of_interest(image):
    mask = np.zeros_like(image)

    triangle = np.array([[
        (750, image_height_bottom),
        (950, image_height_top),
        (1350, image_height_bottom), ]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def show_image_on_plot(image):
    # this function will show captured image to a plot so we can see it in cartesian
    plt.xticks(np.arange(0, 2000, 50.0))
    plt.yticks(np.arange(0, 2000, 50.0))
    plt.imshow(image)
    plt.show()


def lane_detection():
    with mss() as sct:
        monitor = sct.monitors[captured_monitor]
        while True:
            frame = np.array(sct.grab(monitor))
            canny_image = canny(frame)
            # show_image_on_plot(canny_image)
            cropped_canny = region_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
            averaged_lines = average_slope_intercept(lines)
            line_image = display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            cv2.imshow("result", combo_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    lane_detection()
