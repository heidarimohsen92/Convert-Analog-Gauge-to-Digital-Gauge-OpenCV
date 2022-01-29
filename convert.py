import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture('Analog-Gauge.mp4')
frame_size = np.int64((cap.get(3), cap.get(4)))
frame_rate = cap.get(5)
output = cv2.VideoWriter('Digital-Gauge.avi', cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, frame_size)
speed_total = []

while(1):
    ret, frame = cap.read()
    if not ret:
        break
    img_hough_linp = frame.copy()
    frame_gray = cv2.cvtColor(frame, 6)
    frame_edge = cv2.Canny(frame_gray, 100, 200)
    lines = cv2.HoughLinesP(frame_edge, rho=1, theta=np.pi/180, threshold=60, minLineLength=30, maxLineGap=3)
    
    if not (lines is None):
        mag_sel = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ang = (180/np.pi)*np.arctan((y2-y1)/(x2-x1+np.finfo(float).eps))
            ang = -ang

            if min(x2,x1) < 72 or ang < 0:
                    ang += 180

            mag = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if mag > mag_sel:
                    mag_sel = mag
                    ang_sel = ang
                    x1_sel, y1_sel, x2_sel, y2_sel = x1, y1, x2, y2
                 
        
        cv2.line(img_hough_linp, (x1_sel, y1_sel), (x2_sel, y2_sel), (0, 255, 0), 2)
        speed = -1.4*ang_sel+289
        cv2.putText(img_hough_linp, str(int(speed)), (80, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        speed_total.append(speed)
        print(mag_sel, ang_sel, x1_sel, x2_sel)
        
        cv2.imshow("img_hough_linp", img_hough_linp)
        cv2.waitKey(1)
        
    output.write(img_hough_linp)

output.release()
cap.release()