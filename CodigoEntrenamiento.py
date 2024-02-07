#%% Librerias
import cv2
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fcmeans import FCM
from sklearn import svm
from keras.layers import Dense
from keras.models import Sequential
from keras.models import save_model
#from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

#from sklearn.tree import export_text
#from sklearn.tree import plot_tree
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

Colores=['red',
         'gold',
         'lawngreen',
         'deepskyblue',
         'navy',
         'darkviolet']

Labels=['Voluntario 1 (Enfermo)',
        'Voluntario 2 (Enfermo)',
        'Voluntario 3 (Enfermo)',
        'Voluntario 4 (Enfermo)',
        'Voluntario 5 (Sano)',
        'Voluntario 6 (Sano)']
#%% Base de datos 
Vol = [1,2,3,4,5,6]
#%% Datos cada voluntario (arreglo para ser desconcatenado)

TemperaturaVoluntarios=[]
DerivadaVoluntarios=[]
DatosVoluntarios={1:0,2:0,3:0,4:0,5:0,6:0}
PorcentajeRecuperacionVoluntarios=[]
EtiquetasVoluntarios=[1,1,1,1,0,0]
# Diabetes y prediabetes = 1 // Sanos = 0

for Voluntario in Vol:
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
        plt.figure(figsize=(16,8))
        plt.plot(DatosRoi[i*j+i:i*j+j+i+1], c = 'blue')
        plt.grid()
        plt.title("Grafica temperatura en: " + Orden_ROIs[i] + " Voluntario #" + str(Voluntario))
        plt.xlabel("Numero de fotografia")
        plt.ylabel("Temperatura °C")
        plt.show()
        #print(str(i*j+i),' ',(i*j+j+i))
    TemperaturaVoluntarios.append(DatosRoi)
    PorcentajeRecuperacion=[]
    for i in range(10):
        PorcentajeRecuperacion.append(round(100*(DatosRoi[32*i+31]-DatosRoi[32*i+1])/(DatosRoi[i*32]-DatosRoi[32*i+1]),2))
        #print(DatosRoi[i*32] , DatosRoi[32*i+30])
    
    PromedioRecuperacion = np.mean(PorcentajeRecuperacion)
    #%% Derivada
    CambioTemperatura = []
    DatosTraining = 0
    for k in range(len(DatosRoi) - 1):
        if (k+1)%32 != 0:
            #print(DatosRoi[k],DatosRoi[k+1],(2*DatosRoi[k+1]-2*DatosRoi[k]))
            CambioTemperatura.append(Derivada(DatosRoi, i = k))
            
    for i in range(10):        
        plt.figure(figsize=(16,8))
        plt.plot(CambioTemperatura[i*31:i*31+31], c = 'red')        
        plt.grid()
        plt.title("Grafica derivada temperatura en: " + Orden_ROIs[i] + " Voluntario #" + str(Voluntario))
        plt.xlabel("Intervalo de fotografias")
        plt.ylabel("Tasa cambio de temperatura °C/min")
        plt.show()
    DerivadaVoluntarios.append(CambioTemperatura)    
    CambioTemperatura = np.reshape(CambioTemperatura, (len(CambioTemperatura), 1)).T            
    DatosVoluntarios[Voluntario] = CambioTemperatura
    #%% Graficas
    """
    plt.imshow(Base,cmap='gray') 
    plt.title('Img Control voluntario: ' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    plt.imshow(Final,cmap='gray') 
    plt.title('Img Final voluntario: ' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    plt.imshow(BaseBW,cmap='gray') 
    plt.title('Img Control Gris voluntario: ' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    plt.imshow(FinalBW,cmap='gray') 
    plt.title('Img Final Gris voluntario: ' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    plt.imshow(BaseMediana,cmap='gray') 
    plt.title('Imagen Base Filtrada voluntario: ' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    plt.imshow(BaseMediana,cmap='gray') 
    plt.title('Imagen Final Filtrada voluntario: ' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    plt.imshow(THBase,cmap='gray') 
    plt.title('Umbralizada Base voluntario: ' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    plt.imshow(THFinal,cmap='gray') 
    plt.title('Umbralizada Final voluntario: ' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    #plt.imshow(BaseMediana,cmap='gray') 
    #plt.title('Mascara Frascos Amarillos'), plt.xticks([]),plt.yticks([])
    #plt.show()
    
    #plt.imshow(FinalMediana,cmap='gray') 
    #plt.title('Mascara Frascos Amarillos'), plt.xticks([]),plt.yticks([])
    #plt.show()
    
    
    plt.imshow(THBaseMediana,cmap='gray') 
    plt.title('Umbralizada Base filtrada voluntario: ' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    plt.imshow(THFinalMediana,cmap='gray') 
    plt.title('Umbralizada Final filtrada voluntario: ' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    #plt.imshow(DibujoContorno,cmap='gray') 
    #plt.title('Mascara Frascos Amarillos'), plt.xticks([]),plt.yticks([])
    #plt.show()
    
    plt.imshow(DibujoBase,cmap='gray') 
    plt.title('Contornos Base voluntario' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    plt.imshow(DibujoFinal,cmap='gray') 
    plt.title('Contornos final voluntario' + str(Voluntario)), plt.xticks([]),plt.yticks([])
    plt.show()
    
    #cv2.imwrite('Hull1.png', DibujoBase)
    #cv2.imwrite('Hull2.png', DibujoFinal)
    
    # HullBase y HullFinal -> Vertices mano
    # ContornoBase y ContornoFinal -> Mano Normal 
    
    #"""

#%% Concatenacion datos
Datos = 0
for i in Vol:
    
    if i == 1:
        Datos=DatosVoluntarios[i]
    else:
        Datos=np.concatenate((Datos,DatosVoluntarios[i]),axis=0)
        
DatosBarra = np.reshape(Datos, (10*len(Vol),31))    
#%% Graficas Conjuntas

for i in range(len(Orden_ROIs)):
    plt.figure(figsize=(16,8))
    for j in range(len(Colores)):        
        plt.plot(DatosBarra[(j+1)*len(Orden_ROIs)-(len(Orden_ROIs)-i),:], c = Colores[j], label = Labels[j])        
    plt.grid()
    plt.title("Grafica derivada temperatura en: " + Orden_ROIs[i] + " (en todos los voluntarios)")
    plt.xlabel("Intervalo de fotografias")
    plt.ylabel("Tasa cambio de temperatura °C/min")
    plt.legend()
    plt.show()

for i in range(len(Orden_ROIs)):
    plt.figure(figsize=(16,8))
    for j in range(len(TemperaturaVoluntarios)):
        plt.plot(TemperaturaVoluntarios[j][i*32:i*32+32], c = Colores[j], label = Labels[j])
    plt.grid()
    plt.title("Grafica temperatura en: " + Orden_ROIs[i] + " (en todos los voluntarios)")
    plt.xlabel("Intervalo de fotografias")
    plt.ylabel("Temperatura Region de interes °C")
    plt.legend()
    plt.show()

#%% Analisis estadistico de las manos

Estadistica=[]
FotoMin=0
FotoMax=5
for i in range(310): # Numero de datos
    #Punto 1A
    Estadistica.append(['Roi: ' + Orden_ROIs[i//31],
                        'Foto Numero: ' + str(i%31),
                        'Media: ' + str(round(Datos[:,i].mean(axis=0),4)),
                        'Varianza: ' + str(round(np.var(Datos[:,i]),4)),
                        'Minimo' + str(min(Datos[:,i])),
                        'Maximo: ' + str(max(Datos[:,i]))])
    if FotoMin <= i <= FotoMax:
        plt.hist(x=Datos[:,i],  color=Colores[i%6], rwidth=0.99)
        plt.title('Histograma ' + Orden_ROIs[i//32] +' Foto Numero: ' + str(i%32))
        plt.ylabel('Frecuencia')
        plt.show()
        sns.boxplot(Datos[:,i], color=Colores[i%6]) 
        plt.title('Diagrama Caja y Bigotes ' + Orden_ROIs[i//32] +' Foto Numero: ' + str(i%32))
        plt.show() 
        
        
EstadisticaBarra=[]
FotoMinBarra = 0
FotoMaxBarra = 5
for i in range(31): # Numero de datos
    #Punto 1A
    EstadisticaBarra.append(['Foto Numero: ' + str(i%31),
                        'Media: ' + str(round(DatosBarra[:,i].mean(axis=0),4)),
                        'Varianza: ' + str(round(np.var(DatosBarra[:,i]),4)),
                        'Minimo' + str(min(DatosBarra[:,i])),
                        'Maximo: ' + str(max(DatosBarra[:,i]))])
    if FotoMinBarra <= i <= FotoMaxBarra:
        plt.hist(x=DatosBarra[:,i],  color=Colores[i%6], rwidth=0.99)
        plt.title('Histograma Foto Numero: ' + str(i%32))
        plt.ylabel('Frecuencia')
        plt.show()
        sns.boxplot(DatosBarra[:,i], color=Colores[i%6]) 
        plt.title('Diagrama Caja y Bigotes en la foto Numero: ' + str(i%32))
        plt.show() 

#%% Etiquetado
Etiquetas = np.concatenate((np.ones((4, 1)), np.zeros((2,1))),axis=0)
EtiquetasBarra = np.concatenate((np.ones((40, 1)), np.zeros((20,1))),axis=0)
#%% Orden
Index = np.random.permutation(len(Etiquetas))
Index=[3,5,0,2,1,4]
IndexBarra = []
for i in range(len(Index)):
    for j in range(10):
        IndexBarra.append(10*Index[i]+j)
        
Training = Datos[Index[0:4]]
Testing = Datos[Index[4]].reshape((1,310))
Validation =  Datos[Index[5]].reshape((1,310))

Y_Training = Etiquetas[Index[0:4]]
Y_Testing = Etiquetas[Index[4]]
Y_Validation =Etiquetas[Index[5]]

TrainingBarra = DatosBarra[IndexBarra[0:40]]
TestingBarra = DatosBarra[IndexBarra[40:50]]
ValidationBarra =  DatosBarra[IndexBarra[50:60]]

Y_TrainingBarra = EtiquetasBarra[IndexBarra[0:40]]
Y_TestingBarra = EtiquetasBarra[IndexBarra[40:50]]
Y_ValidationBarra = EtiquetasBarra[IndexBarra[50:60]]
#%% Dumificacion etiquetas
Y_Training_Dummies = pd.get_dummies(Y_Training[:,0])
Y_Testing_Dummies = pd.get_dummies(Y_Testing)
Y_Validation_Dummies = pd.get_dummies(Y_Validation)

Y_TrainingBarra_Dummies = pd.get_dummies(Y_TrainingBarra[:,0])
Y_TestingBarra_Dummies = pd.get_dummies(Y_TestingBarra[:,0])
Y_ValidationBarra_Dummies = pd.get_dummies(Y_ValidationBarra[:,0])

#%% Algoritmos Normales  
#"""
#######  Red Neuronal  (ANN)
d=np.size(Training,axis=1)
Red = Sequential() # Se crea un modelo
Red.add(Dense(d, activation = 'sigmoid', input_shape = (d,))) #Capa Entrada
Red.add(Dense(50, activation = 'sigmoid'))                    #Capa Oculta
Caract=np.size(Y_Training_Dummies,axis=1)
Red.add(Dense(2, activation = 'softmax'))                #Capa Salida
# Optimizacion
Red.compile(optimizer = 'adam',
                  loss = 'mean_squared_error', #categorical_crossentropy #mean_squared_error
                  metrics = 'categorical_accuracy')
# Entrenamiento
Red.fit(Training,Y_Training_Dummies, epochs = 250,
              verbose = 1 , workers = 4 , use_multiprocessing=True,
              validation_data = (Validation,Y_Validation_Dummies))

Out_Prob = Red.predict(Testing)
Out_Testing = Out_Prob.round()

Out_Testing = pd.DataFrame(Out_Testing)
Out_Testing = Out_Testing.values.argmax(1)

MatrC_ANN=confusion_matrix(Out_Testing,Y_Testing)


######## Maquina Soporte Vectorial (SVM)
class_weights = {0:1/4, 
                 1:3/4}

SVM = svm.SVC(C = 1, # Default = 1.0
                             gamma = 'auto',
                             degree = 2, #default 2
                             kernel = 'poly',
                             class_weight = class_weights,
                             decision_function_shape = 'ovr',
                             verbose = 1)  # Default 1

SVM.fit(Training, Y_Training)   
# Validación
Y_Out_SVM = SVM.predict(Validation)
MatrC_SVM_V=confusion_matrix(Y_Out_SVM,Y_Validation)
# Prueba
Y_Out_Test_SVM = SVM.predict(Testing)
MatrC_SVM_T=confusion_matrix(Y_Out_Test_SVM,Y_Testing)


######### Arbol descicion DT
Arbol = DecisionTreeClassifier()

Arbol.fit(Training, Y_Training)

Out_Arbol = Arbol.predict(Testing)

Out_Arbol_Valid = Arbol.predict(Validation)

MatrC_DT_T = confusion_matrix(Y_Testing, Out_Arbol)
MatrC_DT_V = confusion_matrix(Y_Validation, Out_Arbol_Valid)
#"""
#%% Valor Añadido (C-means)

#"""
k=2
cmeanModel = FCM(n_clusters=k)
cmeanModel.fit(Datos)
Centroides_C=cmeanModel.centers
DensidadProbs=cmeanModel.u
FCM_Labels=cmeanModel.u.argmax(axis=1)
asignarCmeans=[]
labels = cmeanModel.predict(Datos)
for row in range(np.size(DensidadProbs,axis=0)):
    if DensidadProbs[row,0]<0.2:
        asignarCmeans.append('red')
    elif DensidadProbs[row,0]<0.4:
        asignarCmeans.append('mediumvioletred')
    elif DensidadProbs[row,0]<0.6:
        asignarCmeans.append('darkviolet')
    elif DensidadProbs[row,0]<0.8:
        asignarCmeans.append('mediumslateblue')
    else:
        asignarCmeans.append('blue')

fig = plt.figure()
DatosUsados=[0,31,62]
ax = Axes3D(fig)
ax.scatter(Datos[:, DatosUsados[0]], Datos[:, DatosUsados[1]], Datos[:, DatosUsados[2]], c=asignarCmeans,s=60)
ax.scatter(Centroides_C[:, DatosUsados[0]], Centroides_C[:, DatosUsados[1]], Centroides_C[:, DatosUsados[2]], marker='*', c=['darkblue','darkred'], s=1000)
ax.view_init(185,285)

plt.figure(figsize=(16,8))
plt.scatter(Datos[:, 0], Datos[:, 31], c=asignarCmeans, s=200)
plt.scatter(Centroides_C[:, 0], Centroides_C[:, 31], marker='*', c=['darkblue','darkred'], s=1000)
plt.grid()
plt.xlabel('t1')
plt.ylabel('t2')
plt.title('The Elbow Method showing the optimal k')
plt.legend()
plt.show()
#"""

#%% Guarado Modelos 310 Datos (1 dato por 31 fotos)

save_model(Red, 'ANN310Datos.h5')
joblib.dump(SVM, 'SVM310Datos.pkl')
joblib.dump(Arbol, 'DT310Datos.pkl')
np.save('centroides310Datos.npy', Centroides_C)

#%% Algoritmos por ROI
#"""
#######  Red Neuronal

dBarra=np.size(TrainingBarra,axis=1)
Red2 = Sequential() # Se crea un modelo
Red2.add(Dense(dBarra, activation = 'sigmoid', input_shape = (dBarra,))) #Capa Entrada
Red2.add(Dense(50, activation = 'sigmoid'))                    #Capa Oculta
Caract=np.size(Y_TrainingBarra_Dummies,axis=1)
Red2.add(Dense(2, activation = 'softmax'))                #Capa Salida
# Optimizacion
Red2.compile(optimizer = 'adam',
                  loss = 'mean_squared_error', #categorical_crossentropy #mean_squared_error
                  metrics = 'categorical_accuracy')
# Entrenamiento
Red2.fit(TrainingBarra,Y_TrainingBarra_Dummies, epochs = 250,
              verbose = 1 , workers = 4 , use_multiprocessing=True,
              validation_data = (ValidationBarra,Y_ValidationBarra_Dummies))

Out_Prob_Barra = Red2.predict(TestingBarra)
Out_Testing_Barra = Out_Prob_Barra.round()

Out_Testing_Barra = pd.DataFrame(Out_Testing_Barra)
Out_Testing_Barra = Out_Testing_Barra.values.argmax(1)

MatrC_Barra=confusion_matrix(Out_Testing_Barra,Y_TestingBarra)


######## Maquina Soporte Vectorial (SVM)
class_weights_Barra = {0:10/40, 
                       1:30/40}

SVM_Barra = svm.SVC(C = 1, # Default = 1.0
                             gamma = 'auto',
                             degree = 2, #default 2
                             kernel = 'poly',
                             class_weight = class_weights_Barra,
                             decision_function_shape = 'ovr',
                             verbose = 1)  # Default 1

SVM_Barra.fit(TrainingBarra, Y_TrainingBarra)   
# Validación
Y_Out_SVM_Barra = SVM_Barra.predict(ValidationBarra)
MatrC_SVM_V_Barra = confusion_matrix(Y_Out_SVM_Barra,Y_ValidationBarra)
# Prueba
Y_Out_Test_SVM_Barra = SVM_Barra.predict(TestingBarra)
MatrC_SVM_T_Barra = confusion_matrix(Y_Out_Test_SVM_Barra,Y_TestingBarra)
######### Arbol descicion DT
Arbol_Barra = DecisionTreeClassifier()

Arbol_Barra.fit(TrainingBarra, Y_TrainingBarra)

Out_Arbol_Barra = Arbol_Barra.predict(TestingBarra)

Out_Arbol_Valid_Barra = Arbol_Barra.predict(ValidationBarra)

MatrC_DT_T_Barra = confusion_matrix(Y_TestingBarra, Out_Arbol_Barra)
MatrC_DT_V_Barra = confusion_matrix(Y_ValidationBarra, Out_Arbol_Valid_Barra)
#"""

#%% Valor Añadido (C-means)

#"""
cmeanModelBarra = FCM(n_clusters=k)
cmeanModelBarra.fit(DatosBarra)
Centroides_CBarra=cmeanModelBarra.centers
DensidadProbsBarra=cmeanModelBarra.u
FCM_LabelsBarra=cmeanModelBarra.u.argmax(axis=1)
asignarCmeansBarra=[]
labels = cmeanModelBarra.predict(DatosBarra)
for row in range(np.size(DensidadProbsBarra,axis=0)):
    if DensidadProbsBarra[row,0]<0.2:
        asignarCmeansBarra.append('red')
    elif DensidadProbsBarra[row,0]<0.4:
        asignarCmeansBarra.append('mediumvioletred')
    elif DensidadProbsBarra[row,0]<0.6:
        asignarCmeansBarra.append('darkviolet')
    elif DensidadProbsBarra[row,0]<0.8:
        asignarCmeansBarra.append('mediumslateblue')
    else:
        asignarCmeansBarra.append('blue')

fig = plt.figure()
DatosUsadosBarra=[0,1,2]
ax = Axes3D(fig)
ax.scatter(DatosBarra[:, DatosUsadosBarra[0]], DatosBarra[:, DatosUsadosBarra[1]], DatosBarra[:, DatosUsadosBarra[2]], c=asignarCmeansBarra,s=60)
ax.scatter(Centroides_CBarra[:, DatosUsadosBarra[0]], Centroides_CBarra[:, DatosUsadosBarra[1]], Centroides_CBarra[:, DatosUsadosBarra[2]], marker='*', c=['darkblue','darkred'], s=1000)
ax.view_init(185,285)

plt.figure(figsize=(16,8))
plt.scatter(DatosBarra[:, 0], DatosBarra[:, 30], c=asignarCmeansBarra, s=200)
plt.scatter(Centroides_CBarra[:, 0], Centroides_CBarra[:, 30], marker='*', c=['darkblue','darkred'],  s=1000)
plt.grid()
plt.xlabel('t1')
plt.ylabel('t2')
plt.title('The Elbow Method showing the optimal k')
plt.legend()
plt.show()
#"""

#%% Guarado Modelos 31 Datos (10 datos en 31 fotos)

save_model(Red2, 'ANN31Datos.h5')
joblib.dump(SVM_Barra, 'SVM31Datos.pkl')
joblib.dump(Arbol_Barra, 'DT31Datos.pkl')
np.save('centroides31Datos.npy', Centroides_CBarra)

#%% Validacion Cruzada
#ValidacionCruzada=0
#ValidacionCruzadaBarra=0
#%% Pregunta por estadistica    
print('\014')
print('¿desea ver alguna estadistica en concreto?\n')
print('Presione 1 para las estadisticas de los datos completos (10 datos x 32 fotos)')
print('Presione 2 para las estadisticas de los datos por ROI (1 dato x 31 fotos)')
print('Presione otra cosa para salir')
while (a := input('Opción: ')):
    if a=='1':
        print('Dijite el numero de fotografia entre 0 y 30')
        b = input('# Fotografia: ')
        print('Dijite el numero de Region de interes entre las siguientes: ')
        for j in range(len(Orden_ROIs)):
            print( str(j) + '. ' + Orden_ROIs[j])
        c = input('# Region de interes: ')
        print('\014')
        print('Considere que lo denominado fotografia es en realidad el cambio entre esa fotografia y la siguiente')
        for i in range(6):
            print(Estadistica[31*int(c)+int(b)][i])
        print('Unidades: °C/min')
    elif a=='2':
        print('Dijite el numero de fotografia entre 0 y 30')
        b = input('# Fotografia: ')
        print('\014')
        print('Considere que lo denominado fotografia es en realidad el cambio entre esa fotografia y la siguiente')
        for i in range(5):
            print(EstadisticaBarra[int(b)][i])
        print('Unidades: °C/min')
    else:
        break
    print('\n\n')
    print('Desea Ver otras estadisticas')
    print('Presione 1 para las estadisticas de los datos completos (10 datos x 32 fotos)')
    print('Presione 2 para las estadisticas de los datos por ROI (1 dato x 31 fotos)')
    print('Presione otra cosa para salir')
print('Gracias')
