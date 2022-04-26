from confRed import *
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import cv2

#torch.save(model,'modelo.pt')
model = torch.load('./Modelo para imagenes grandes/modelo_0.0358.pt',map_location='cpu')


device = "cuda" if torch.cuda.is_available() else "cpu"
imagen=torchvision.io.read_image('./Imagenes/imagen248.jpg')
#imagen= np.load('./Imagenes/imagen181.npy')
imagen=imagen.float()/255


#model.eval()
#with torch.no_grad():
 #   output = model(imagen.unsqueeze(0).to(device))[0]
  #  mascara_pred = torch.argmax(output, axis=0)

#np.save('./mascara',mascara_pred.numpy())
#print(imagen.shape)
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,10))
#ax1.imshow(imagen.squeeze(0).permute(1,2,0))
#ax2.imshow(mascara_pred.squeeze().cpu().numpy())
#plt.show()

#imagen=cv2.read_image('./Imagenes/imagen248.jpg',0)
output = model(imagen.unsqueeze(0))[0]
mascara = torch.argmax(output,axis=0).numpy().astype(np.uint8)

contornos,_=cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mascara,contornos,-1,(2,2,2),1)

areaMayor=0
ContornoMayor=None
for cnt in contornos:
    area = cv2.contourArea(cnt)
    if area>areaMayor:
        areaMayor=area
        contornoMayor=cnt
                    
print(areaMayor)
   
if areaMayor>100:
    m = cv2.moments(contornoMayor)
    if m['m00'] != 0:
        cx = int(m['m10']/m['m00'])
        cy = int(m['m01']/m['m00'])
        cv2.drawContours(mascara, [contornoMayor], -1, (2,2,2), 2)
        cv2.circle(mascara, (cx, cy), 3, 2, 1)
    print(cx)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,10))
ax1.imshow(imagen.squeeze(0).permute(1,2,0))
ax2.imshow(mascara.squeeze())
plt.show()