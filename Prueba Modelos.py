#%% Librerias
import cv2
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cdist
from random import randint

numero_aleatorio = randint(1, 999)

#%%
print('\014')
#from FuncionesTDG import *
#%% Funciones
def Leer_DB(Ruta, Formato,Nombre,h = 500,w = 290, Resize=False):
    """ 
    Inmporta una base de datos de imagenes desde una carpeta del PC 
    Paramtros: Ruta y Formato
    Ruta: Ruta de la carpeta que contiene las imagenes
    Formato: Formato de todas las imagenes
    Nombre: Nombre DB
    h, w dimensiones imagenes (default = 500 x 290)
    """
    import pathlib
    import cv2
    Imagen = pathlib.Path(Ruta)
    Formato="*."+ str(Formato)
    Base = {
        Nombre:list(Imagen.glob(Formato))
        }
    X = []
    for label, images in Base.items():
        for image in images:
            img = cv2.imread(str(image)) # Reading the image
            if img is not None:
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if (Resize == True):
                    img = cv2.resize(img, (h, w))
                X.append(img)
            else:
                print(f"No se pudo leer {str(image)}")
    return X

def IdentificacionROI(Imagen,h,w,areaRoi,CteProp=1,Tmin=0,Umbral=0):
    """
    Parameters
    ----------
    Imagen : TYPE
        DESCRIPTION. Entra una imagen en BGR
    h : TYPE Entero
        DESCRIPTION. Pixel H de la ROI
    w : TYPE Entero
        DESCRIPTION. Pixel W de la ROI
    areaRoi : TYPE Entero
        DESCRIPTION. Area cuadrado de la ROI
    CteProp : TYPE float
        DESCRIPTION. Contante para convertir de color a °C o °F
        Default = 1/ grados/180
    Returns
    -------
    TYPE Float
        DESCRIPTION. Devuelve el color medio/ Temp Media

    """
    global H
    H=Imagen[h-areaRoi:h+areaRoi+1,w-areaRoi:w+areaRoi+1]
    H=np.setdiff1d(H,0)
    H=max(H)    
    return (H-Umbral)*CteProp+Tmin

def Derivada(Muestra,DeltaMuestras = 1, DeltaTiempo = 0.5, Cte = 1, i = 0):
    """
    Parameters
    ----------
    Muestra : TYPE
        Temperatura t = n
    DeltaMuestras : TYPE Entero
        Intervalo entre muestras
    DeltaTiempo : TYPE Entero
        Tiempo entre muestras 0.5 minutos 
    Cte : TYPE float
        Constante aumento valor
        Default = 1
    i : TYPE float
        Numero de la muestra
        Default = 0
    Returns
    -------
    TYPE Float
        DESCRIPTION. Devuelve el color medio/ Temp Media

    """
    #Calculo = (Muestra[i+DeltaMuestras]-Muestra[i])
    return Cte*(Muestra[i+DeltaMuestras]-Muestra[i])/DeltaTiempo

#%% Variables Globales
TopeAnMed = [255 , 270]
TopePulIn = [150 , 370]
Delta = 110
DeltaUltimaROI = 40

Centro=[TopeAnMed[0] , TopePulIn[1] + Delta]
RoiCentro = [TopeAnMed[0] + DeltaUltimaROI, TopePulIn[1] + Delta//2]
#%% Datos cada voluntario (arreglo para ser desconcatenado)
# Diabetes y prediabetes = 1 // Sanos = 0
Voluntario = 6

DataBase = Leer_DB('D:\Tesis\DB\Voluntario ' + str(Voluntario) + '\Termografia\Serie', 'jpg', 'V1Termo')
Base=DataBase[0]
Final=DataBase[-1]

#%% pre-procesamiento
# Umbralizada
BaseBW = cv2.cvtColor(Base, cv2.COLOR_RGB2GRAY)
FinalBW = cv2.cvtColor(Final, cv2.COLOR_RGB2GRAY)

# Se filtra la imagen (doble filtrado)	
BaseMediana = cv2.medianBlur(BaseBW,9)
BaseMediana = cv2.medianBlur(BaseMediana,5)
FinalMediana = cv2.medianBlur(FinalBW,9)
FinalMediana = cv2.medianBlur(FinalMediana,5)

# Umbralizado con y sin filtro
ret,THBase = cv2.threshold(BaseBW,50,255,cv2.THRESH_BINARY)
ret,THFinal = cv2.threshold(FinalBW,50,255,cv2.THRESH_BINARY)
ret,THBaseMediana = cv2.threshold(BaseMediana,50,255,cv2.THRESH_BINARY)
ret,THFinalMediana = cv2.threshold(FinalMediana,50,255,cv2.THRESH_BINARY)

# Contornos
ContornoBase, hierarchy1 = cv2.findContours(THBaseMediana, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ContornoFinal, hierarchy2 = cv2.findContours(THFinalMediana, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Se crean los contornos con base en las vertices
hullBase = []
hullFinal = []
# Calculo puntos por vertice
for i in range(len(ContornoBase)):
    hullBase.append(cv2.convexHull(ContornoBase[i], False))
for i in range(len(ContornoFinal)):
    hullFinal.append(cv2.convexHull(ContornoFinal[i], False))

# Se crea la mascara de 0 para los vertices y bordes
DibujoBase = np.zeros((THBaseMediana.shape[0], THBaseMediana.shape[1], 3), np.uint8)
DibujoFinal = np.zeros((THFinalMediana.shape[0], THFinalMediana.shape[1], 3), np.uint8)

# draw contours and hull points
for i in range(len(ContornoBase)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(DibujoBase, ContornoBase, i, color_contours, 1, 8, hierarchy1)
    # draw ith convex hull object
    cv2.drawContours(DibujoBase, hullBase, i, color, 1, 8)
for i in range(len(ContornoFinal)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(DibujoFinal, ContornoFinal, i, color_contours, 1, 8, hierarchy2)
    # draw ith convex hull object
    cv2.drawContours(DibujoFinal, hullFinal, i, color, 1, 8)

# Identificacion vertices de la mano
Max_Base=0
Max_Final=0
for i in range(len(ContornoBase)):
    if np.size(ContornoBase[i]) > np.size(ContornoBase[Max_Base]):
        Max_Base = i
for i in range(len(ContornoFinal)):
    if np.size(ContornoFinal[i]) > np.size(ContornoFinal[Max_Final]):
        Max_Final = i

# Extraccion Vertice de las manos         
ManoBase = ContornoBase[Max_Base]
ManoFinal = ContornoFinal[Max_Final]

# Se estandarizan los vertices para dejarlos en formato nx2 en lugar de nx1x2 
VerticesBase = hullBase[Max_Base]
VerticesBase = np.reshape(VerticesBase,(len(VerticesBase),2))
VerticesFinal = hullFinal[Max_Final]
VerticesFinal = np.reshape(VerticesFinal,(len(VerticesFinal),2))

#  Variables con las puntas de los dedos
VerticesManoBase = []
VerticesManoFinal = []
# Delta de filtrado de vertices de los dedos
Th = 30 # Px
# Se identifican los puntos extremos de los 5 dedos
for i in range(len(VerticesBase)):
    if ((VerticesBase[i-1][0] <= (VerticesBase[i][0] + Th)) and ((VerticesBase[i][0] - Th) <= VerticesBase[i-1][0]))  and  ((VerticesBase[i-1][1] <= (VerticesBase[i][1] + Th)) and ((VerticesBase[i][1] - Th) <= VerticesBase[i-1][1])):
        pass
    else:
        if VerticesBase[i][1] < 500:
            VerticesManoBase.append(VerticesBase[i][:])
for i in range(len(VerticesFinal)):
    if ((VerticesFinal[i-1][0] <= (VerticesFinal[i][0] + Th)) and ((VerticesFinal[i][0] - Th) <= VerticesFinal[i-1][0]))  and  ((VerticesFinal[i-1][1] <= (VerticesFinal[i][1] + Th)) and ((VerticesFinal[i][1] - Th) <= VerticesFinal[i-1][1])):
        pass
    else:
        if VerticesFinal[i][1] < 500:
            VerticesManoFinal.append(VerticesFinal[i][:])


VerticesManoBase2 = []
VerticesManoFinal2 = []
# Delta de filtrado de vertices de los dedos
Th = 30 # Px
# Se identifican los puntos extremos de los 5 dedos
for i in range(0,-len(VerticesBase),-1):
    if ((VerticesBase[i][0] <= (VerticesBase[i+1][0] + Th)) and ((VerticesBase[i+1][0] - Th) <= VerticesBase[i][0]))  and  ((VerticesBase[i][1] <= (VerticesBase[i+1][1] + Th)) and ((VerticesBase[i+1][1] - Th) <= VerticesBase[i][1])):
        pass
    else:
        if VerticesBase[i][1] < 500:
            VerticesManoBase2.append(VerticesBase[i][:])
for i in range(0,-len(VerticesFinal),-1):
    if ((VerticesFinal[i][0] <= (VerticesFinal[i+1][0] + Th)) and ((VerticesFinal[i+1][0] - Th) <= VerticesFinal[i][0]))  and  ((VerticesFinal[i][1] <= (VerticesFinal[i+1][1] + Th)) and ((VerticesFinal[i+1][1] - Th) <= VerticesFinal[i][1])):
        pass
    else:
        if VerticesFinal[i][1] < 500:
            VerticesManoFinal2.append(VerticesFinal[i][:])
            
    
# Variables rois de las venas entre dedos          
OtrasROIsBase = []
OtrasROIsFinal = []  

# Se ordenan los vertices por dedo (Pulgar-Indice-Medio-Anular-Meñique)
VerticesManoBase = sorted(VerticesManoBase, key=lambda x: x[0])
VerticesManoFinal = sorted(VerticesManoFinal, key=lambda x: x[0])

VerticesManoBase2 = sorted(VerticesManoBase2, key=lambda x: x[0])
VerticesManoFinal2 = sorted(VerticesManoFinal2, key=lambda x: x[0])
for i in range(len(VerticesManoBase)):
    VerticesManoBase[i] = (VerticesManoBase[i] + VerticesManoBase2[i])//2 
    VerticesManoFinal[i] =  (VerticesManoFinal[i] + VerticesManoFinal2[i])//2 
# Se calcula el punto medio de los dedos consecutivos
for i in range(len(VerticesManoBase) - 1):
    OtrasROIsBase.append((VerticesManoBase[i][:] + VerticesManoBase[i+1][:])/2)
for i in range(len(VerticesManoFinal) - 1):
    OtrasROIsFinal.append((VerticesManoFinal[i][:] + VerticesManoFinal[i+1][:])/2)

# Se inicializan los porcentajes de proximidad con el centro de la mano y de los puntos previos     
PorcentajeROIFalanje = 0.80   
PorcentajeROIVena = 0.4
RoiDedosFinal=[]   
RoiDedosBase=[]   

# Calculo rois Falanjes
for i in range(len(VerticesManoBase)):
    a=[int(PorcentajeROIFalanje*VerticesManoBase[i][0] + (1 - PorcentajeROIFalanje)*Centro[0]), int(PorcentajeROIFalanje*VerticesManoBase[i][1] + (1 - PorcentajeROIFalanje)*Centro[1])]
    RoiDedosBase.append(a)
for i in range(len(VerticesManoFinal)):
    a=[int(PorcentajeROIFalanje*VerticesManoFinal[i][0] + (1 - PorcentajeROIFalanje)*Centro[0]), int(PorcentajeROIFalanje*VerticesManoFinal[i][1] + (1 - PorcentajeROIFalanje)*Centro[1])]
    RoiDedosFinal.append(a)

# Calculo rois Venas 
for i in range(len(OtrasROIsBase)):
    a=[int(PorcentajeROIVena*OtrasROIsBase[i][0] + (1 - PorcentajeROIVena)*Centro[0]), int(PorcentajeROIVena*OtrasROIsBase[i][1] + (1 - PorcentajeROIVena)*Centro[1])]
    RoiDedosBase.append(a)
for i in range(len(OtrasROIsFinal)):
    a=[int(PorcentajeROIVena*OtrasROIsFinal[i][0] + (1 - PorcentajeROIVena)*Centro[0]), int(PorcentajeROIVena*OtrasROIsFinal[i][1] + (1 - PorcentajeROIVena)*Centro[1])]
    RoiDedosFinal.append(a)

# Conexion todas las roi's
RoiDedosBase.append(RoiCentro)
RoiDedosFinal.append(RoiCentro)

# Incializa vector Datos
Datos=[]

#%% Procesamiento 
# Extraccion datos y grafica de la temperatura por roi

for i in range(len(DataBase)):
    DataFrame = pd.read_csv('D:\Tesis\DB\Voluntario ' +str(Voluntario) + '\CSV\V' +str(Voluntario) + 'F' + str(i) + '.csv',skiprows = 2, header = None, decimal = ',', sep = ';')
    DataFrame = DataFrame.to_numpy()
    if i == 0:
        for j in range(len(RoiDedosBase)):
            Datos.append(IdentificacionROI(DataFrame,RoiDedosBase[j][1],RoiDedosBase[j][0],3))
    else:
        for j in range(len(RoiDedosFinal)):
            Datos.append(IdentificacionROI(DataFrame,RoiDedosFinal[j][1],RoiDedosFinal[j][0],3))
        
DatosRoi=[]
for i in range(10):
    for j in range(len(Datos)//10):        
        DatosRoi.append(Datos[10*j+i])

#%% Derivada
CambioTemperatura = []
DatosTraining = 0
for k in range(len(DatosRoi) - 1):
    if (k+1)%32 != 0:
        #print(DatosRoi[k],DatosRoi[k+1],(2*DatosRoi[k+1]-2*DatosRoi[k]))
        CambioTemperatura.append(Derivada(DatosRoi, i = k))
        


CambioTemperatura = np.reshape(CambioTemperatura, (len(CambioTemperatura), 1)).T            


#%% Datos
Datos = CambioTemperatura      
DatosBarra = np.reshape(Datos, (10,31))    
#%% Importar Modelos
modelo_red = load_model('ANN310Datos.h5')
modelo_svm = joblib.load('SVM310Datos.pkl')
modelo_arbol = joblib.load('DT310Datos.pkl')
centroides = np.load('centroides310Datos.npy')

modelo = ['Red Neuronal',
          'Maquina Soporte vectorial',
          'Arbol Descicion']
#%% Prediccion Modelo
prediccionANN = modelo_red.predict(Datos)
prediccionANN = prediccionANN.round()
prediccionSVM = modelo_svm.predict(Datos)
prediccionDT = modelo_arbol.predict(Datos)
distancias = cdist(Datos, centroides)

#%% Validacion Cruzada
Modelos=[ prediccionANN[0][1], prediccionSVM[0], prediccionDT[0] ]
ValidacionCruzada = np.mean(Modelos)
texto=[]

for i in range(3):
    if Modelos[i] == 1:        
        print('Segun el algoritmo: ' + modelo[i] + ' Se encuentra Enfermo')
        texto.append('Segun el algoritmo: ' + modelo[i] + ' Se encuentra Enfermo')
    else:
        print('Segun el algoritmo: ' + modelo[i] + ' Se encuentra Sano')
        texto.append('Segun el algoritmo: ' + modelo[i] + ' Se encuentra Sano')
    texto.append('\n')

print('Su puntuaje acorde a los modelos es del: ' + str(round(100*(1-ValidacionCruzada),2)) + '%')
texto.append('Su puntuaje acorde a los modelos es del: ' + str(round(100*(1-ValidacionCruzada),2)) + '%')
texto.append('\n\n\n')
#ValidacionCruzadaBarra=0
print('\n\n')
#%% Importar Modelos Por ROI
modelo_red_barra = load_model('ANN31Datos.h5')
modelo_svm_barra = joblib.load('SVM31Datos.pkl')
modelo_arbol_barra = joblib.load('DT31Datos.pkl')
centroides_barra = np.load('centroides31Datos.npy')

Orden_ROIs=['Dedo Pulgar',
            'Dedo Indice',
            'Dedo Medio',
            'Dedo Anular',
            'Dedo Meñique',
            'Venas MC Entre dedos Pulgar e indice',
            'Venas MC Entre dedos indice e Medio',
            'Venas MC Entre dedos Medio e Anular',
            'Venas MC Entre dedos Anular e Meñique',
            'Metacarpianos']

#%% Prediccion Modelo _barra
prediccionANN_barra = modelo_red_barra.predict(DatosBarra)
prediccionSVM_barra = modelo_svm_barra.predict(DatosBarra)
prediccionDT_barra = modelo_arbol_barra.predict(DatosBarra)
distancias_barra = cdist(DatosBarra, centroides_barra)

#%% Validacion 
Modelos_barra=[]
for i in range(10):
    Modelos_barra.append(prediccionANN_barra[i][1].round())
    Modelos_barra.append(prediccionSVM_barra[i])
    Modelos_barra.append(prediccionDT_barra[i])
ValidacionCruzadaBarra = np.mean(Modelos_barra)
for i in range(10):
    
    if (Modelos_barra[3*i] +  Modelos_barra[3*i+1] +  Modelos_barra[3*i+2])/3 <= 0.25:
        print('Segun Los modelos la ROI: ' + Orden_ROIs[i] + ' Se encuentra Saludable')
        texto.append('Segun Los modelos la ROI: ' + Orden_ROIs[i] + ' Se encuentra Saludable')        
    elif (Modelos_barra[3*i] +  Modelos_barra[3*i+1] +  Modelos_barra[3*i+2])/3 <= 0.5:
        print('Segun Los modelos la ROI: ' + Orden_ROIs[i] + ' Es probalble que cuente con riesgo moderado')
        texto.append('Segun Los modelos la ROI: ' + Orden_ROIs[i] + ' Es probalble que cuente con riesgo moderado')
    elif (Modelos_barra[3*i] +  Modelos_barra[3*i+1] +  Modelos_barra[3*i+2])/3 <= 0.75:
        texto.append('Segun Los modelos la ROI: ' + Orden_ROIs[i] + ' Es probalble que cuente con riesgo elevado')
        print('Segun Los modelos la ROI: ' + Orden_ROIs[i] + ' Es probalble que cuente con riesgo elevado')
    else:
        texto.append('Segun Los modelos la ROI: ' + Orden_ROIs[i] + ' Se encuentra Enferma')
        print('Segun Los modelos la ROI: ' + Orden_ROIs[i] + ' Se encuentra Enferma')
    texto.append('\n')   
print('Su puntuaje acorde a los modelos es del: ' + str(round(100*(1-ValidacionCruzadaBarra),2)) + '%')
texto.append('Su puntuaje acorde a los modelos es del: ' + str(round(100*(1-ValidacionCruzadaBarra),2)) + '%')   


archivo = open("EstadoPersona_" + str(numero_aleatorio) +".txt", "w")
for i in texto:
    archivo.write(i)
archivo.close()
