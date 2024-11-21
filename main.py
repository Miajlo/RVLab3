import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

FLANN_INDEX_KDTREE = 1

def trim(frame):
    while not np.sum(frame[0]):  # Check if the top row is all zeros
        frame = frame[1:]  # Remove the first row
    while not np.sum(frame[-1]):  # Check if the bottom row is all zeros
        frame = frame[:-1]  # Remove the last row
    while not np.sum(frame[:, 0]):  # Check if the left column is all zeros
        frame = frame[:, 1:]  # Remove the first column
    while not np.sum(frame[:, -1]):  # Check if the right column is all zeros
        frame = frame[:, :-1]  # Remove the last column
    return frame

if __name__ == '__main__':
    img1 = cv.imread("image1.JPG")
    img2 = cv.imread("image2.JPG")
    img3 = cv.imread("image3.JPG")

    detector = cv.SIFT_create()

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    output1 = cv.warpPerspective(
        img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    output1[0:img1.shape[0], 0:img1.shape[1]] = img1
    output1 = trim(output1)

    kp1, des1 = detector.detectAndCompute(output1, None)
    kp2, des2 = detector.detectAndCompute(img3, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    output2 = cv.warpPerspective(
        img3, M, (output1.shape[1] + img3.shape[1], output1.shape[0]))

    final_image = np.zeros(
        [output1.shape[0], output1.shape[1]+img3.shape[1], 3], dtype=np.uint8)
    final_image[0:output1.shape[0], 0:output1.shape[1]] = output1

    final_image[0:output1.shape[0], output1.shape[1] -
                                    10:] = output2[0:, output1.shape[1]-10:]
    final_image = trim(final_image)
    final_image = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)
    plt.imshow(final_image)
    plt.show()
    cv.imwrite("result.JPG", final_image)

