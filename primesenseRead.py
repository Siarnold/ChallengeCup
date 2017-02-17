#!/usr/bin/python3
# primesenseRead tested on Ubuntu System
# Type 's' to save the images in current directory
# with name 'color/depth_[CURRENR_TIME]'.
# Required to copy link-library 'openni2' to directory 'primesense'
# Copyleft(c), Liu Jiang, Tsinghua University

from __future__ import print_function
from primesense import openni2
from primesense.openni2 import *
from primesense._openni2 import *
import numpy as np
import cv2
import time

openni2.initialize()

d = openni2.Device.open_any()
print (d.get_device_info())

print ("IR", d.get_sensor_info(openni2.SENSOR_IR))
print ("DEPTH", d.get_sensor_info(openni2.SENSOR_DEPTH))
print ("COLOR", d.get_sensor_info(openni2.SENSOR_COLOR))

depth = d.create_depth_stream()
depth.set_property(ONI_STREAM_PROPERTY_VIDEO_MODE, OniVideoMode(pixelFormat = OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX = 640, resolutionY = 480, fps = 30))
color = d.create_color_stream()
color.set_property(ONI_STREAM_PROPERTY_VIDEO_MODE, OniVideoMode(pixelFormat = OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 640, resolutionY = 480, fps = 30))

assert depth is not None
print ("v-fov:", depth.get_vertical_fov())
print ("h-fov:", depth.get_horizontal_fov())
print ("camera settings:", depth.camera)

depth.start()
color.start()

n=1
while(1):
    s = openni2.wait_for_any_stream([depth], 2)
    if not s:
        continue
    frame = depth.read_frame()
    frame2=color.read_frame()
    
    if frame.videoMode.pixelFormat not in (openni2.PIXEL_FORMAT_DEPTH_100_UM, openni2.PIXEL_FORMAT_DEPTH_1_MM):
        print ("Unexpected frame format", frame.videoMode.pixelFormat)
        continue

    data = frame.get_buffer_as_uint16()
    data = np.reshape(data, (480, -1))
    data=((data-data.min())/(data.max()-data.min())*255).astype(np.uint8)
    cv2.imshow("depth",data)
    
    data2 = frame2.get_buffer_as_uint8()
    data2 = np.reshape(data2, (480, -1, 3))
    data2 = cv2.cvtColor(data2, cv2.COLOR_RGB2BGR)
    cv2.imshow("color",data2)
    k= cv2.waitKey(20)
    k = chr(k%128)
    if k == 'q':
        break
    elif k == 's':
        n=n+1
        name="color_"+str(time.localtime())+".jpg"
        cv2.imwrite(name, data2)
        
color.stop()
depth.stop()
openni2.unload()


