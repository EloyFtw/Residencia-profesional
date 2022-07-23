import cv2

cap = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(0)
print("Inicio")	

def tomarFoto(ix):
	leido, frame = cap.read()
	leido1, frame2 = cap1.read()
	print("Tomada")
	if leido == True and leido1 == True:
		cv2.imwrite("./fotos/foto"+str(ix)+".png", frame)
		cv2.imwrite("./fotos/foto"+str((ix+1))+".png", frame2)
		print("Foto guardada correctamente")
	else:
		print("Error al acceder a la cámara")

i=0
while True:
	input("Tomar Foto:")
	tomarFoto(i)
	i=i+2


cap1.release()
cap.release()
