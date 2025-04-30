#!/usr/bin/env python3
from os import TMP_MAX
import cv2
import numpy as np
import sys


def detect_edges(image):
  """Find edge points in a grayscale image.

  Args:
  - image (2D uint8 array): A grayscale image.

  Return:
  - edge_image (2D float array): A heat map where the intensity at each point
      is proportional to the edge magnitude.
  """

  sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=int)
  sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=int)

  def conv(image, kernel):
    # with padding!
    if (kernel.shape[0]!=kernel.shape[1]):
        raise ValueError("Invalid convolution kernel: the shape is not a square.")
    if (kernel.shape[0]%2==0):
        raise ValueError("Invalid convolution kernel: the height and the width should be an odd number.")
    k = kernel.shape[0]
    result = np.zeros(image.shape)
    input = image.copy()
    input = np.pad(input, ((int(k/2),int(k/2)),(int(k/2),int(k/2))), 'reflect')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i,j] = (input[i:i+k,j:j+k]*kernel).sum()
    return result

  G_x = conv(image, sobel_x)
  G_y = conv(image, sobel_y)
  edge_image = np.abs(G_x)+np.abs(G_y)

  return edge_image


def hough_circles(edge_image, edge_thresh, radius_values):
  """Threshold edge image and calculate the Hough transform accumulator array.

  Args:
  - edge_image (2D float array): An H x W heat map where the intensity at each
      point is proportional to the edge magnitude.
  - edge_thresh (float): A threshold on the edge magnitude values.
  - radius_values (1D int array): An array of R possible radius values.

  Return:
  - thresh_edge_image (2D bool array): Thresholded edge image indicating
      whether each pixel is an edge point or not.
  - accum_array (3D int array): Hough transform accumulator array. Should have
      shape R x H x W.
  """
  
  thresh_edge_image = edge_image.copy()
  thresh_edge_image = np.array(thresh_edge_image>edge_thresh, dtype=bool)
  accum_array = np.zeros((len(radius_values),edge_image.shape[0],edge_image.shape[1]))
  for i in range(edge_image.shape[0]):
    for j in range(edge_image.shape[1]):
        if (thresh_edge_image[i][j]):
            for r_i in range(len(radius_values)):
                r = radius_values[r_i]
                for k in range(max(0,i-r),min(i+r,edge_image.shape[0])):
                    delta = int(np.round(np.sqrt(r**2-(k-i)**2)))
                    if (j-delta>=0):
                        accum_array[r_i][k][j-delta] += 1
                    if (delta>0) and (j+delta<edge_image.shape[1]):
                        accum_array[r_i][k][j+delta] += 1

  return thresh_edge_image, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh, nms=2):
  """Find circles in an image using output from Hough transform.

  Args:
  - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
      original color image instead of its grayscale version so the circles
      can be drawn in color.
  - accum_array (3D int array): Hough transform accumulator array having shape
      R x H x W.
  - radius_values (1D int array): An array of R radius values.
  - hough_thresh (int): A threshold of votes in the accumulator array.

  # ======= MY VERSION ======= #
  - nms (int): An argument I personally added.
      [nms<=0] Do not apply Non-Maximum Suppression.
      [nms>0]  Apply Non-Maximum Suppression in the (nms*2+1)*(nms*2+1)-
            neighborhood. Only the local maximum in the (nms*2+1)*(nms*2+1)-
            neighborhood will remain after NMS.
      nms=0,1,2 are recommended.
  # === END of MY VERSION === #

  Return:
  - circles (list of 3-tuples): A list of circle parameters. Each element
      (r, y, x) represents the radius and the center coordinates of a circle
      found by the program.
  - circle_image (3D uint8 array): A copy of the original image with detected
      circles drawn in color.
  """

  circles = []
  circle_array = accum_array > hough_thresh
  
  if (nms>0):
    accum_nms = accum_array.copy()
    accum_nms[accum_array <= hough_thresh] = 0

    for r_i in range(len(radius_values)):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if (accum_nms[r_i][i][j]>0):
                    local = accum_nms[r_i-nms:r_i+nms,i-nms:i+nms,j-nms:j+nms]
                    if (accum_nms[r_i][i][j]!=local.max()):
                        accum_nms[r_i][i][j] = 0
                    else:
                        local = np.array(local>=local.max(), dtype=int)
                        if (local.sum()>1):
                            accum_nms[r_i][i][j] = 0
    
    circle_array = accum_nms > 0

  circle_image = image
  for r_i in range(len(radius_values)):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (circle_array[r_i][i][j]):
                r = radius_values[r_i]
                circles.append((r,j,i))
                circle_image = cv2.circle(circle_image, (j,i), r, color=(0,255,0), thickness=2)
  
  return circles, circle_image


def main(argv):
  img_name = argv[0]
  edge_thresh = int(argv[1])
  vote_thresh = int(argv[2])
  r_min = int(argv[3])
  r_max = int(argv[4])
  if (r_max<r_min):
    tmp = r_min
    r_min = r_max
    r_max = tmp
  if (len(argv)>5):
    nms = int(argv[5])
  else:
    nms = 0

  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  radius = list(range(r_min, r_max+1))
  labeled_image = detect_edges(gray_image)
  labeled_image, accum_array = hough_circles(labeled_image, edge_thresh, radius)
  circles, circle_image = find_circles(gray_image, accum_array, radius, vote_thresh, nms)
  labeled_image = np.array(labeled_image, dtype=int)*255
  cv2.imwrite('output/' + img_name + '_edges.png', labeled_image)
  cv2.imwrite('output/' + img_name + '_circles.png', circle_image)
  print(len(circles))


if __name__ == '__main__':
  main(sys.argv[1:])