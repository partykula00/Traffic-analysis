# Traffic-analysis
Traffic analysis and modeling based on video data

This repository contains traffic detection and tracking scripts focused on highway traffic.
The detection uses ready-made models trained on the COCO 2017 collection downloaded from TensorFlow 2 Detection Model Zoo:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md


For tracking, a simple algorithm based on centroids and shortest distance is used
A mapping of real coordinates to pixel coordinates is used for speed estimation.

Scheme of operation of the program:
![image](https://github.com/partykula00/Traffic-analysis/assets/107066853/e6974f65-47e5-4006-80c2-df69db5d9712)

Video ouput screen:
![program_output](https://github.com/partykula00/Traffic-analysis/assets/107066853/9d1ce913-f761-4ddb-a270-0ec9862eaa29)



