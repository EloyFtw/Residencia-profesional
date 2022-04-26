from confRed import *
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import cv2

#torch.save(model,'modelo.pt')
model = torch.load('./Modelo para imagenes pequeñas/modelo.pt',map_location='cpu')
#model = torch.load('./Modelo para imagenes pequeñas/modelo_0.0326.pt',map_location='cpu')



device = "cuda" if torch.cuda.is_available() else "cpu"
imagen=torchvision.io.read_image('./ImagenesEscaladas/imagen291.jpg')
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


#Aqui calculo los bordes.
i=0
bordes  = np.zeros([10, 240,320]);
areas  = np.zeros(10);
perimetros  = np.zeros(10);


while i < 240:
    j=0
    while j < 320:
        x=mascara[i][j]
        if mascara[i][j] != 0:
            areas[x]= areas[x]+ 1; #En esta linea se calcula el area de cada pixel de cada objeto

            if (i-1) > 0 : 
                if mascara[i][j] != mascara[i-1][j]:                
                    bordes[x][i][j] = x
                    perimetros[x]= perimetros[x]+ 1  #En esta linea se calcula el perimetro de cada pixel de cada objeto
            if (i+1) < 240 : 
                if mascara[i][j] != mascara[i+1][j]:
                    bordes[x][i][j] = x
                    perimetros[x]= perimetros[x]+ 1 
            if (j-1) > 0 : 
                if mascara[i][j] != mascara[i][j-1]:
                    bordes[x][i][j] = x
                    perimetros[x]= perimetros[x]+ 1 
            if (j+1) < 320 : 
                if mascara[i][j] != mascara[i][j+1]:
                    bordes[x][i][j] = x
                    perimetros[x]= perimetros[x]+ 1

        j = j+1        
    i=i+1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,10))

#Guardo los bordes en un arreglo 10, uno por cada objeto.
#print(mascara.shape)
n=0
bords  = np.zeros([240,320]); #Esta variable almacena en sus 10 posiciones los 10 objetos de interes

while n<10:
    bords += bordes[n] 
    print("Area:",str(areas[n]))
    print("Perimetro:",(str(perimetros[n])))
    n = n+1
# Aqui los junto para mostrarlos

ax1.imshow(mascara.squeeze())

ax2.imshow(bords.squeeze())
plt.show()


puntoMy = np.zeros(240)
puntoMx = np.zeros(320)

while i < 240: #Alto
    j=0
    cont = 0
    while j < 320: #Ancho
        if bordes[i][j] == mascara[i][j]:
            cont = cont + 1
        j = j+1       
         
    i=i+1

#print(mascara)
#contornos, jerarquia = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
'''
contornos,_= cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(mascara,contornos,  ,(0,0,0),1)
print(len(contornos))
cnt=contornos[0]
m=cv2.moments(cnt)
#print(m)
#if m['m00'] != 0:
cx = int(m['m10']/m['m00'])
cy = int(m['m01']/m['m00'])

cv2.circle(mascara, (cx, cy), 5, (0, 255,0), -1)
'''

'''
cv2.imshow("mascara", mascara)
cv2.waitKey(0)
cv2.destroyAllWindows()

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
'''
'''
for cnt in contornos:
    area = cv2.contourArea(cnt)
    print(area)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,10))
ax1.imshow(imagen.squeeze(0).permute(1,2,0))
ax2.imshow(mascara.squeeze())
plt.show()
''' 


