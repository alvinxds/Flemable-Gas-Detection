# Flamable Gas Detection
The following scripts were conducted within the framework of the master thesis entitled "Automatic Detection of Flammable Gases (C3H8, CH4) in Thermal Infrared Data Video".

The main idea behind this research was the developement of a gas detection system based on computer vision and machine learning techniques, utilizing thermal image data. 
The infrared and thermal data can be further analyzed for the construction of a variety of systems, such as an early detection system of leaks at oil refineries or detection of high voltage transmission at substations.

This research was focused on designing an automatic gas detection and visualization system, addresing two individual approaches. The first relies on a machine learning techniques, training and evaluating different classifiers such as SVM and Adaboost from thermal image data. Within this context, two different training samples were created from the thermal videos displaying leaks of flemable gases. The first one was created from selected two-dimensional gas and non-gas image patches, whereas the second one was three-dimensional gas and non-gas patches containg the parameter of time (2d-image patches 48x48 pixels extracted for 11 frames).

From the created training samples, different feature descriptors were evaluated such as LBP, Edge Orientation Histogram,  HOGHOF, LBPTOP, which were also used for the training of the abovementioned classifiers. 

The second approach relies on the fusion of thermal and true color data. To be more specific, rgb were simultaneously acqueried jointly with the thermal videos. This system relies on background extraction techniques after the sychronization of the thermal and rgb videos, in order to detect the flamable gases at the pixel level.

The master thesis can be found in greek here: https://dspace.lib.ntua.gr/xmlui/handle/123456789/49984

The following videos are the results of this analysis.


 <p align="center">
  <b>mlapapa</b>
  <b></b>
  <img src="https://user-images.githubusercontent.com/39597223/124764689-e56c2100-df3d-11eb-8682-adcecfea27d4.gif" width="320" height="240" >
 </p>

https://user-images.githubusercontent.com/39597223/124756507-30ce0180-df35-11eb-9d43-50ce533b2870.mp4

https://user-images.githubusercontent.com/39597223/124758275-13019c00-df37-11eb-8efb-df680c161c80.mov

https://user-images.githubusercontent.com/39597223/124758353-26146c00-df37-11eb-88d8-f4944dec0d9f.mov



