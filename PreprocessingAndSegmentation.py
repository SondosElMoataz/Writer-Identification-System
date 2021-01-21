import cv2
import numpy as np


def Preprocess(image):
     
    # Remove salt and pepper noise    
    # Remove noise 
    median = cv2.medianBlur(image,5)
    blur = cv2.GaussianBlur(image,(5,5),0)
   
    greyImg = image
    
    # Otsu's Binarization
    ret3,img = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   
    # Remove header and footer
    length, width = img.shape
    up, down, left, right = 0, length - 1, 0, width - 1

    minWidthOfLines = width/2
    yes,contours,no = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    offsetHeader = 20
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if w < minWidthOfLines:
            continue
        if y < length // 2:
            up = max(up, y + offsetHeader)
        else:
            down = min(down, y - offsetHeader)

    offset = 30
    left = left + offset
    right = right -offset
    noHeaderAndFooter = img[up:down + 1, left:right + 1]
    noHeaderAndFooter = np.asarray(noHeaderAndFooter)
    
    noHeaderAndFooterGrey = greyImg[up:down + 1, left:right + 1]
    noHeaderAndFooterGrey = np.asarray(noHeaderAndFooterGrey) 
  

    # To crop the image
    row, col = noHeaderAndFooter.shape
    tolerance = 15

    sumOfRows = np.sum(noHeaderAndFooter, axis = 1)
    rowIndices = np.where(sumOfRows< (col-tolerance)*255)
    up = np.min(rowIndices)
    down = np.max(rowIndices)

    sumOfColoumns = np.sum(noHeaderAndFooter, axis = 0)
    colIndices = np.where(sumOfColoumns< (row-tolerance)*255)
    left = np.min(colIndices)
    right = np.max(colIndices)

    binarized = noHeaderAndFooter[up:down + 1, left:right + 1]
    binarized = np.asarray(binarized)
    
    greyscale = noHeaderAndFooterGrey[up:down + 1, left:right + 1]
    greyscale = np.asarray(greyscale)
 

    # Segmentation of Lines
    rowIndicesShifted = np.roll(rowIndices, -1)
    rowIndicesShifted = rowIndicesShifted[0]

    transitionIndices = np.where(np.abs(rowIndices - rowIndicesShifted) > 10)
    transitionIndices = transitionIndices[1]

    rowIndices = rowIndices[0]

    downIndices= rowIndices[transitionIndices]

    transitionIndicesUp = np.insert(transitionIndices,0,-1)
    transitionIndicesUp = np.delete(transitionIndicesUp,-1)

    upIndices= rowIndices[transitionIndicesUp+1]
    
    segmentsBinarized = []
    segmentsGrey = []
    totalSize=0
    whiteSpaceTolerance=0.97
    for i in range(transitionIndices.shape[0]):
        currSegment= noHeaderAndFooterGrey[upIndices[i]:downIndices[i] + 1, left:right + 1]
        if((np.sum(currSegment))<(currSegment.shape[0]*currSegment.shape[1]*whiteSpaceTolerance*255)):
            segmentsBinarized.append(noHeaderAndFooter[upIndices[i]:downIndices[i] + 1, left:right + 1])
            segmentsGrey.append(currSegment)
          
    segmentsGrey=np.asarray(segmentsGrey)
    
    return greyscale, binarized, segmentsBinarized, segmentsGrey