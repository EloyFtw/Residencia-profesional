import os
import json
import numpy as np
import cv2

def colorear(clase, regions ):

   if clase==regions['region_attributes']['clase']:
      polygons = regions['shape_attributes']

      if clase=='mesa': R,V,A=  [128,0,0]
      elif clase=='vaso': R,V,A=  [0,0,128]
      elif clase=='tasa': R,V,A=  [128,128,0]
      elif clase=='plato': R,V,A=  [128,0,128]
      elif clase=='celular': R,V,A=  [128,128,128]
      else : R,V,A= 0,0,0
   
   
      if (polygons['name']=='polygon'):
           countOfPoints = len(polygons['all_points_x'])
           points = [None] * countOfPoints
           for i in range(countOfPoints):
              x = int(polygons['all_points_x'][i])
              y = int(polygons['all_points_y'][i])
              points[i] = (x, y)
      elif (polygons['name']=='rect'):
           points = [None] * 4
           x = int(polygons['x'])
           y = int(polygons['y'])
           w = int(polygons['width'])
           h = int(polygons['height'])
           points[0] = (x,y)
           points[1] = (x+w,y)
           points[2] = (x+w,y+h)
           points[3] = (x,y+h)
      contours = np.array(points)
      for i in range(width):
         for j in range(height):
            if cv2.pointPolygonTest(contours, (i, j), False) > 0:
               maskImage[j,i,0] = A
               maskImage[j,i,1] = V
               maskImage[j,i,2] = R


IMAGE_FOLDER = "./Imagenes/"
MASK_FOLDER = "./Mascaras/"
PATH_ANNOTATION_JSON = './MRD.json'
annotations = json.load(open(PATH_ANNOTATION_JSON, 'r'))
imgs = annotations["_via_img_metadata"]
for imgId in imgs:
    filename = imgs[imgId]['filename']
    regions = imgs[imgId]['regions']
    image_path = os.path.join(IMAGE_FOLDER, filename)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    maskImage = np.zeros((height,width,3), dtype=np.uint8)
    
    for i in range(len(regions)): colorear('mesa', regions[i]) 
    for i in range(len(regions)): colorear('plato', regions[i]) 
    for i in range(len(regions)): colorear('vaso', regions[i])
    for i in range(len(regions)): colorear('tasa', regions[i])
    for i in range(len(regions)): colorear('celular', regions[i])
                 
    savePath = MASK_FOLDER + filename.split('.')[0]+'.png'
    cv2.imwrite(savePath, maskImage)


