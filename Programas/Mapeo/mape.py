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
#imagen=torchvision.io.read_image('./ImagenesEscaladas/imagen1.jpg')
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


while i < 240:#Alto
    j=0
    while j < 320: #Ancho
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


#Mandar el nombre de caja objeto
def NombreObjeto(valor):
    if(valor==0):
        return "Fondo"
    if(valor==1):
        return "Conector"        
    if(valor==2):
        return "Celular"
    if(valor==3):
        return "Plato" 
        #return "Conector"        
    if(valor==4):
        return "Vaso"
    if(valor==5):
        return "Mesa" 
    if(valor==6):
        return "Persona"
    if(valor==7):
        return "Herramienta" 
    if(valor==8):
        return "Garrafon"
    if(valor==9):
        return "Obstaculo" 
    else:
        return "0"

def ponerTexto(texto, ubicacion):
    font1 = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(mascara, texto, ubicacion, font1, .8, 1, 1)

#Guardo los bordes en un arreglo 10, uno por cada objeto.
#print(mascara.shape)
n=1
bords  = np.zeros([240,320]); #Esta variable almacena en sus 10 posiciones los 10 objetos de interes

while n<10:
    bords += bordes[n]
    #print(NombreObjeto(n) + "  Area: ",str(areas[n]) + "  Perimetro: ",(str(perimetros[n])))
    n = n+1
# Aqui los junto para mostrarlos

##Punto medio
puntoM = np.zeros([10,2,2])

listEjeX = np.zeros([4,2])

n=0
while n<10:
    i=0
    listEjeX = np.zeros([4,2])
    while i < 240:#Alto
        j=0
        while j < 320: #Ancho
            if bordes[n][i][j] != 0:
                listEjeX[0]= [i,j]
            j = j+1 

        j=319
        while j > 0: #Ancho
            if bordes[n][i][j] != 0:
                listEjeX[1]= [i,j]
            j = j - 1    
        
        i=i+1

    i=239
    while i > 0:#Alto
        j=0
        while j < 320: #Ancho
            if bordes[n][i][j] != 0:
                listEjeX[2]= [i,j]        
                
            j = j+1 

        j=319
        while j > 0: #Ancho
            if bordes[n][i][j] != 0:
                listEjeX[3]= [i,j]
            j = j - 1    
        
        i=i-1

    #print(listEjeX)
    cy = int((listEjeX[0][0] + listEjeX[1][0] + listEjeX[2][0] + listEjeX[3][0]) / 4)
    cx = int((listEjeX[0][1] + listEjeX[1][1] + listEjeX[2][1] + listEjeX[3][1]) / 4)

    cv2.circle(bords, (cx, cy), 3, 2, 1)
    puntoM[n] = [cx,cy]
    ponerTexto(NombreObjeto(n), (cx,cy))
    n = n + 1

#print(puntoM)

cv2.circle(bords, (160, 120), 4, 2, -1) #Centro de la imagen
#ponerTexto("0,0", (160,120))


Coord = np.zeros([10,2])
i = 0
while i<10:
    if puntoM[i][0][0] != 0:
        if puntoM[i][0][0] == 160:
            Coord[i][0]= 0
        elif puntoM[i][0][0] > 160:
            Coord[i][0]= puntoM[i][0][0] - 160
        elif puntoM[i][0][0] < 160:
            Coord[i][0]= puntoM[i][0][0] - 160
    i=i+1

i = 0
while i<10:
    if puntoM[i][0][1] != 0:
        if puntoM[i][0][1] == 120:
            Coord[i][1]= 0
        elif puntoM[i][0][1] > 120:
            Coord[i][1]= puntoM[i][0][0] - 120
        elif puntoM[i][0][1] < 120:
            Coord[i][1]= puntoM[i][0][0] - 120
    i=i+1
#Dibujar la imagen
#print() #Aqui se guardan las coordenadas
#####
def CalcularDistancia(area):
    dis= np.log(area)
    return dis;

n=1
while n<10:
    print(NombreObjeto(n) + "  Area: ",str(areas[n]) + "  Perimetro: ",(str(perimetros[n])) + "    Coordenadas x, y  " + str(Coord[n]))
    print(CalcularDistancia(areas[n]))
    n = n+1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,10))

ax1.imshow(mascara.squeeze())

ax2.imshow(bords.squeeze())
plt.show()


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


