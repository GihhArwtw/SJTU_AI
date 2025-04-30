#!/usr/bin/env python3
import cv2
import numpy as np
import sys


def binarize(gray_image, thresh_val):
  # TODO: 255 if intensity >= thresh_val else 0

  binary_image = np.array(gray_image>thresh_val, dtype=int) * 255
  return binary_image

def label(binary_image):
  # TODO

  size = binary_image.shape
  labeled_image = np.zeros(size, dtype=int)

  cnctcomp = [0]
  num_comp = 0

  def find(x):        # Disjoint Set Union
    if (x>=len(cnctcomp)):
      raise ValueError("Exceeding the length of the current Disjoint Set Union.")
    if (cnctcomp[x]==x): return x
    cnctcomp[x] = find(cnctcomp[x])
    return cnctcomp[x]

  # _____________________________________________
  # FIRST PASS
  labeled_image[-1][0] = binary_image[-1][0]
  for j in range(1,size[1]):
    if (binary_image[-1][j] == 0):
      labeled_image[-1][j] = 0
    elif (binary_image[-1][j-1]>0):
      labeled_image[-1][j] = labeled_image[-1][j-1]
    else:
      num_comp += 1
      cnctcomp += [num_comp]
      labeled_image[-1][j] = num_comp

  for i in range(size[0]-2,-1,-1):
    for j in range(size[1]):
      if (binary_image[i][j] == 0):
        labeled_image[i][j] = 0
      else:
        if (labeled_image[i+1][j-1]>0):
          labeled_image[i][j] = labeled_image[i+1][j-1]
        elif (labeled_image[i+1][j]+labeled_image[i][j-1]==0):
          num_comp += 1
          cnctcomp += [num_comp]
          labeled_image[i][j] = num_comp
        else:
          x = labeled_image[i+1][j]
          y = labeled_image[i][j-1]
          if (binary_image[i+1][j]+binary_image[i][j-1]>255):     # both are "1".
            labeled_image[i][j] = x
            if (cnctcomp[x]!=cnctcomp[y]):
              x = find(cnctcomp[x])
              y = find(cnctcomp[y])
              cnctcomp[max(x,y)] = min(x,y)
          else:
            labeled_image[i][j] = x+y              # since either x or y is 0.

  # ____________________________________
  # label color assignment
  colors = []
  for i in range(1,len(cnctcomp)):
    cnctcomp[i] = find(i)
    if (cnctcomp[i]==i): colors.append(i)
  cnctcomp = [0]+[int((colors.index(x)+1)/float(len(colors))*255) for x in cnctcomp[1:]]
  
  # ____________________________________
  # SECOND PASS
  for i in range(size[0]):
    for j in range(size[1]):
      labeled_image[i][j] = cnctcomp[labeled_image[i][j]]

  return labeled_image

def get_attribute(labeled_image):
  # TODO

  size = labeled_image.shape
  image = labeled_image.copy()
  directions = [[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1]]
  attribute_list = []

  for i in range(size[0]-1,-1,-1):
    for j in range(size[1]):
      if (image[i][j]>0):
        head = 0
        tail = 0
        queue = [[j,i]]
        image[i][j] = -image[i][j]
        while (head<=tail):
          [x,y] = queue[head]
          for k in range(8):
            xx = x+directions[k][0]
            yy = y+directions[k][1]
            if (xx<0)or(xx>=size[1])or(yy<0)or(yy>=size[0]):  continue
            if (image[yy][xx]>0):
              tail += 1
              queue += [[xx,yy]]
              image[yy][xx] = -image[yy][xx]              
          head += 1

        queue = np.array(queue).transpose()
        queue[1] = size[0] - queue[1]            # note that the origin is on the left bottom.
        x_bar = queue[0].mean()
        y_bar = queue[1].mean()
        queue = queue - np.array([[x_bar],[y_bar]])
        a = (queue[0]**2).sum()
        b = (queue[0]*queue[1]).sum()*2.
        c = (queue[1]**2).sum()

        def second_moment(theta):
          return a*(np.sin(theta)**2)-b*np.sin(theta)*np.cos(theta)+c*(np.cos(theta)**2)

        pos = dict([('x',x_bar),('y',y_bar)])
        if np.abs(a-c)<1e-16:
          orient = np.pi/4.
        else:
          orient = np.arctan(float(b)/float(a-c)) / 2.
        moment1 = second_moment(orient)
        moment2 = second_moment(orient+np.pi/2.)
        if (moment1>moment2):
          tmp = moment1
          moment1 = moment2
          moment2 = tmp
          orient += np.pi/2
        round = moment1 / float(moment2)

        attribute_list += [dict([('position',pos),('orientation',orient),('roundedness',round)])]

  return attribute_list

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labeled_image = label(binary_image)
  attribute_list = get_attribute(labeled_image)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
  print(attribute_list)

if __name__ == '__main__':
  main(sys.argv[1:])
