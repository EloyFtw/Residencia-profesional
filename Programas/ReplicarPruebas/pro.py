import serial
import time
import torch
import numpy as np
import torchvision
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import threading
import warnings

def conv3x3_bn(ci, co):
    return torch.nn.Sequential(
        torch.nn.Conv2d(ci, co, 3, padding=1),
        torch.nn.BatchNorm2d(co),
        torch.nn.ReLU(inplace=True)
    )

class deconv(torch.nn.Module):
    def __init__(self, ci, co):
        super(deconv, self).__init__()
        self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
        self.conv1 = conv3x3_bn(ci, co)
        self.conv2 = conv3x3_bn(co, co)
    
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX, 0, diffY, 0))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class out_conv(torch.nn.Module):
    def __init__(self, ci, co, coo):
        super(out_conv, self).__init__()
        self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
        self.conv = conv3x3_bn(ci, co)
        self.final = torch.nn.Conv2d(co, coo, 1)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX, 0, diffY, 0))
        x = self.conv(x1)
        x = self.final(x)
        return x

class UNetResnet(torch.nn.Module):
    def __init__(self, n_classes=3, in_ch=3):
        super().__init__()

        self.encoder = torchvision.models.resnet18(pretrained=False) # Para predicción únicamente
        self.deconv1 = deconv(512,256)
        self.deconv2 = deconv(256,128)
        self.deconv3 = deconv(128,64)
        self.out = out_conv(64, 64, n_classes)

    def forward(self, x):
        x_in = x.clone().detach()
        x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x = self.encoder.layer4(x3)
        x = self.deconv1(x, x3)
        x = self.deconv2(x, x2)
        x = self.deconv3(x, x1)
        x = self.out(x, x_in)
        return x

warnings.filterwarnings('ignore')
device="cuda"
print("Pytorch con Soporte para Cuda:",torch.cuda.is_available())

puerto_serial = serial.Serial(
    port="/dev/ttyACM0",
    baudrate=115200,
    timeout=0.1
)

time.sleep(0.1)

class HiloCamara(threading.Thread):
    def __init__(self,camara):
        threading.Thread.__init__(self)
        self.camara=camara
        self.ultima_imagen=None
        self.activo=True

    def run(self):
        while self.activo:
            ret,self.ultima_imagen=self.camara.read()
            time.sleep(0.01)
        print("Cerrando Hilo Cámara")

'''
class HiloSerial(threading.Thread):
    def __init__(self,puerto_serial):
        threading.Thread.__init__(self)
        self.puerto_serial=puerto_serial
        self.activo=True

    def run(self):
        while self.activo:
            try:
                dato=puerto_serial.readline()
                print(dato)
            except: None
        print("Cerrando Hilo Serial")
'''

dispW=320
dispH=240
flip=2
camSet='nvarguscamerasrc sensor-id=0 sensor-mode=5 ! nvvidconv ! video/x-raw(memory:NVMM), width=320, height=240, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
camara=cv2.VideoCapture(camSet)
camara.set(cv2.CAP_PROP_FRAME_WIDTH,320)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
camara.set(cv2.CAP_PROP_BUFFERSIZE,1)

#hiloSerial=HiloSerial(puerto_serial)
#hiloSerial.start()

hiloCamara=HiloCamara(camara)
hiloCamara.start()

modelo = torch.load('modelo.pt')
modelo.to(device)
print("Modelo cargado")
modelo.eval()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(30,10))
img=ax1.imshow(np.zeros((240,320)))
mas=ax2.imshow(np.zeros((240,320)))

AREA_MAX=30000

try:
    while True:
        #pausa=input("Pausa:")
        with torch.no_grad():
            if hiloCamara.ultima_imagen is not None:
                inicio=time.time()
                imagen=hiloCamara.ultima_imagen
                imagen=cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
                imagen_torch=torch.from_numpy(imagen).permute(2,0,1).float()/255
                imagen_cuda=imagen_torch.unsqueeze(0).to(device)
                output = modelo(imagen_cuda)[0]
                mascara = torch.argmax(output,axis=0)
                mascara = mascara.cpu().numpy().astype(np.uint8)
                contornos,_=cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                areaMayor=0
                contornoMayor=None
                for cnt in contornos:
                    area = cv2.contourArea(cnt)
                    if area>areaMayor:
                        areaMayor=area
                        contornoMayor=cnt                    
                if areaMayor>0:
                    m = cv2.moments(contornoMayor)
                    if m['m00'] != 0:
                        cx = int(m['m10']/m['m00'])
                        #cy = int(m['m01']/m['m00'])
                        #cv2.drawContours(mascara, [contornoMayor], -1, (2,2,2), 2)
                        #cv2.circle(mascara, (cx, cy), 3, 2, 1)
                else:
                    areaMayor=-1
                    cx=-1
                    #cy=-1
                print(areaMayor)   
                print(cx)
                if areaMayor!=-1:
                    cadena="<area>"+str(areaMayor)+"<cx>"+str(cx)+"<fin>\n"
                else:
                    cadena="<area>"+str(AREA_MAX)+"<cx>"+str(160)+"<fin>\n"
                puerto_serial.write(cadena.encode())
                fin=time.time()
                print(fin-inicio)
                mascara = np.eye(3)[mascara]
                #img.set_data(imagen)
                #mas.set_data(mascara)
                #fig.canvas.draw_idle()
                #plt.pause(0.00001)
                #time.sleep(0.01)
                print("--------------------")
except KeyboardInterrupt:
    cadena="<area>"+str(AREA_MAX)+"<cx>"+str(160)+"<fin>\n"
    puerto_serial.write(cadena.encode())
    #hiloSerial.activo=False
    hiloCamara.activo=False
    camara.release()
    puerto_serial.close()
    #cv2.destroyAllWindows()
finally:
    print("Cerrando Principal")
    pass

