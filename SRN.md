# Selective Refinement Network for High Performance Face Detection

[link](https://arxiv.org/pdf/1809.02693.pdf)

- Single-shot face detector (selective refinement network)
- two step classification and regression operations into an anchor-based face detector to reduce false positives and 
location accuracy
- SRN consists of 2 modules: Selective Two-step Classification (STC) module and the Selective Two-step Regression (STR) module
- STC aims to filter out simple negative anchors from low level detection layers to reduce search space for subsequent classifier
- STR is designed to coarsely adjust the locations and sizes of anchors from high level detection layers to provide better initialization
for the subsequent regressor
- They design a Receptive Field Enhancement (RFE) block to provide more diverse receptive field, helping to better capture faces in extreme
poses
- Their goal is to solve recall efficiency: the number of false positives needs to be reduced at high recall rates
- location accuracy of bounding box locations
- Average precision (AP) at present is high, but precision is not high enough at high recall rates
* For RetinaNet the precision is only about 50% (half of detections are false positives) when the recall rate is equal to 90, which they
define as low recall efficiency. 
* The reason being is that existing algorithms pay more attention to high recall rate but ignore excessive false positives
* To detect tiny images e.g. less than 16x16 pixels, it is necessary to tile plenty of small anchors over the image
* This improves recall yet cause extreme class imbalance problem, leading to excessive false positives
* Research tries to solve with R-CNN like detectors to adress class imbalance by two-stage cascade and sampling heuristics 
* As for single-shot detectors, RetinaNet proposes focal loss to focus training on a sparse set of hard examples and down-weight the loss
assigned to well-classified exampels
* They show that different IoU thresholds, as IoU increases AP drops drastically, indicating the accuracy of bbox location needs improving
* Cascade R-CNN addresses this issue by cascading R-CNN with different IoU thresholds
