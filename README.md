# Flamable Gas Detection
The following scripts were conducted within the framework of the master thesis entitled "Automatic Detection of Flammable Gases (C3H8, CH4) in Thermal Infrared Data Video".

The main idea behind this research was the developement of a gas detection system based on computer vision and machine learning techniques, utilizing thermal image data. 
The infrared and thermal data can be further analyzed for the construction of a variety of systems, such as an early detection system of leaks at oil refineries or detection of high voltage transmission at substations.

This research was focused on designing an automatic gas detection and visualization system, addresing two individual approaches. The first relies on a machine learning techniques, training and evaluating different classifiers such as SVM and Adaboost from thermal image data. Within this context, two different training samples were created from the thermal videos displaying leaks of flemable gases. The first one was created from selected two-dimensional gas and non-gas image patches, whereas the second one was three-dimensional gas and non-gas patches containg the parameter of time (2d-image patches 48x48 pixels extracted for 11 frames).

From the created training samples, different feature descriptors were evaluated such as LBP, Edge Orientation Histogram,  HOGHOF, LBPTOP, which were also used for the training of the abovementioned classifiers. 

The second approach relies on the fusion of thermal and true color data. To be more specific, rgb were simultaneously acqueried jointly with the thermal videos. This system relies on background extraction techniques after the sychronization of the thermal and rgb videos, in order to detect the flamable gases at the pixel level.

The master thesis can be found in greek here: https://dspace.lib.ntua.gr/xmlui/handle/123456789/49984

https://user-images.githubusercontent.com/39597223/123999519-4abe9000-d9db-11eb-9180-ff81311f272f.mp4





https://user-images.githubusercontent.com/39597223/124751338-2c064f00-df2f-11eb-8c41-d6b7c90c28a2.mp4








