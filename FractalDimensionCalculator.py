import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class FractalDimensionCalculator:
    
    def __init__(self, image):
        
        self.image = image
        if(image.ndim == 3):
            self.colorImage = True
        else:
            self.colorImage = False
        
    
    def computeBoxFractalDimension(self, boxLength):
        imageWidth = np.size(self.image, 0)
        imageHeight = np.size(self.image, 1)
        
        nStepsWidth =imageWidth//boxLength
        nStepsHeight = imageHeight//boxLength
        
        scale = imageWidth/boxLength
        N = 0
        for i in range(0,nStepsHeight):
            index1 = int(i*boxLength)
            for j in range(0,nStepsWidth):
                index2 = int(j*boxLength)
                subImageSum = -1
                #Imagen de 2 dimensiones
                if(self.colorImage == False):
                    subImage = self.image[index1:index1+boxLength-1, index2:index2 + boxLength-1]
                    subImageSum = sum(sum(subImage))
                #Imagen de 1 dimension 
                else:
                    subImage = self.image[index1:index1 + boxLength-1, index2:index2 + boxLength-1,:]
                    #print(np.size(subImage))
                    subImageSum = sum(sum(sum(subImage)))
                
                if(subImageSum != 0):
                    N = N + 1

        dimension = np.log(N)/np.log(scale)
        
        return N, dimension
    
    def getFractalDimsForDifferentBoxSizes(self, maxBoxSize):
        boxSizes = np.floor(np.flip(np.arange(2, maxBoxSize, 1)))
        nArray = np.zeros(len(boxSizes))
        fractalDimensions = np.zeros(len(boxSizes))
        for i in range(0,len(boxSizes)):
            boxSize = int(boxSizes[i])
            N, fractalDimension = self.computeBoxFractalDimension(boxSize)
            nArray[i] = N
            fractalDimensions[i] = fractalDimension
        
        return boxSizes, nArray, fractalDimensions
    
    def getFractalDimension(self,maxBoxSize, plotRegression):
        boxSizes, nArray, fractalDimensions = self.getFractalDimsForDifferentBoxSizes(maxBoxSize)
        logBoxSizes = np.log(boxSizes)
        logN = np.log(nArray)
        
        a, b, r, p, stdErr= stats.linregress(logBoxSizes, logN)
        minX = np.min(logBoxSizes)
        maxX = np.max(logBoxSizes)
        minY = np.min(logN)
        maxY = np.max(logN)
        
        if(plotRegression):
            yReg = a*logBoxSizes + b
            plt.figure()
            plt.plot(logBoxSizes, logN, marker = "o", color = "#de1919", linestyle = "none")
            plt.plot(logBoxSizes, yReg, color = "#5441ff")
            plt.text((minX + maxX)/2, (minY + maxY)/2, "y = "+str(round(a, 2)) + "x" + " + "+str(round(b, 2)))
            plt.xlabel("Logaritmo del tamaño de caja")
            plt.ylabel("Logaritmo del número de cajas")
        return -a
