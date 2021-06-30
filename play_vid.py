# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:19:05 2019

@author: anton
"""
import cv2

######Set functions
def play_vid(names,window_titles):
    cap = [cv2.VideoCapture(i) for i in names]
    
    frames = [None] * len(names);
    gray = [None] * len(names);
    ret = [None] * len(names);
    
    while True:
        for i,c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read();
        
        for i,f in enumerate(frames):
            if ret[i] is True:
                gray[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                cv2.imshow(window_titles[i], gray[i]);
        
        if cv2.waitKey(80) & 0xFF == ord('q'):
            break
        
    for c in cap:
        if c is not None:
            c.release();
            
    cv2.destroyAllWindows() 