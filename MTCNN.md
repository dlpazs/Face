# Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

These are just my own personal notes not the official, see link below for the official:

[link](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
[link](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)

Abstract: Face detection and alignment in unconstrained environment are challenging due to various poses, illuminations and occlusions. 
Recent studies show that deep learning approaches can achieve impressive performance on these two tasks. 
In this paper, we propose a deep cascaded multi-task framework which exploits the inherent correlation between detection and 
alignment to boost up their performance. In particular, our framework leverages a cascaded architecture with three stages of 
carefully designed deep convolutional networks to predict face and landmark location in a coarse-to-fine manner. In addition, 
we propose a new online hard sample mining strategy that further improves the performance in practice. Our method achieves superior 
accuracy over the state-of-the-art techniques on the challenging FDDB and WIDER FACE benchmarks for face detection, and AFLW 
benchmark for face alignment, while keeps real time performance.

- most previous face detection/alignment methods ignore the correlation between these two tasks

## Pipeline

![alt text](https://kpzhang93.github.io/MTCNN_face_detection_alignment/support/index.png)

- Firstly: Resize (112, 112)
- Stage 1: P-Net -> NMS & Bounding box regression
- Stage 2: R-Net -> NMS & Bounding box regression
- Stage 3: O-Net -> NMS & Bounding box regression
- [Link](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/614a7c42aae8946c7ad4c36b53290860f6256441/2-Figure1-1.png)

- Pipeline of cascaded framework that has a 3 stage multi-task deep convolutional network. Firstly, candidate windows are produced through a fast Proposal Network (P-Net). Then they refind these candidates through a Refinement Network (R-Net). Thirdly, the Output Network (O-Net) produces the final bounding box and facial landmarks
- They propose a new cascaded CNN's framework for joint face detection and alignment, and lightweight CNN for real-time performance
- Method for online hard sample mining

- **Stage 1** P-Net obtains candidate facial windows and bounding box regression vectors. Candidates are calibrated based on the estimated bounding box regression vectors. Then NMS is applied to merge highly overlapped candidates

|CNN|300×ForwardPropagation|Validation Accuracy|
|------|---------------------|--------------------|
|12-Net|0.038s|94.4%|
|P-Net|0.031s|94.6%|
|24-Net|0.738s|95.1%|
|R-Net|0.458s|95.4%|
|48-Net|3.577s|93.2%|
|O-Net|1.347s|95.4% |

- **Stage 2** All candidates few to R-Net, which rejects a large number of false candidates, performs calibration with bounding box regression and conducts NMS

- **Stage 3** Same as stage 2 but identifying face regions with more supervision, outputting 5 facial landmarks

