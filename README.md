# Face
Face Detection, Face Alignment, Landmark detection, Face Recognition

********************************************************

## Facial Recognition

Facial recognition is the task of making a **positive identification* of a face in a photo or video image against a pre-**existing database of faces**. It begins with **detection** - distinguishing human faces from other objects in the image - and then works on **identification** of those detected faces.


### RMDL (Olivetti Faces 5 images) [paper](https://arxiv.org/pdf/1805.01890v2.pdf), [code](https://github.com/kk7nc/RMDL)

Random Multimodel Deep Learning, a new ensemble approach for classification. DL models have achieved SOTA across many domains. RMDL solves the problem of finding the best DL structure/architecture. RMDL accepts many data types from text, video, images.

### FaceNet [paper](https://arxiv.org/pdf/1503.03832.pdf), [code](https://github.com/davidsandberg/facenet)

Implementing face verification and recognition at scale present serious challenges to current approaches. FaceNet learns a mapping from face images to compact Euclidean space where distances correspond to measure of face similarity. The DCNN is trained to optimize the embedding itself, using triplets of faces. They introduce the idea of harmonic embeddings and harmonic triplet loss which describes different versions of face embeddings produced by different networks that allow for direct comparison between each other. 

### GhostVLAD (IJB-A/B) [paper](https://arxiv.org/pdf/1810.09951v1.pdf)

To learn a compact representation of image sets for template-based face recognition. Netowkr which aggregates and embeds the face descriptors produced by deep CNN's into a compact fixed-length representation. This representation requires minimal memory storage and enables efficient similarity computation. Propose novel GhostVLAD layer that includes ghost clusters that don't contribute to aggregation. Quality weighting on input emerges automatically such that informative images contribute more than those with low quality, and ghost clusters enhace network's ability to deal with poor quality images. SOTA IJB-B.

### Honourable Mentions
- ArcFace, Dense U-Nets

********************************************************

## Face Detection

Face detection is the task of detecting faces in a photo or video (and distinguishing them from other objects).

### RetinaFace (WIDER Face Hard) [paper](https://arxiv.org/pdf/1905.00641v2.pdf), [code](https://github.com/deepinsight/insightface)

### AlnnoFace (WIDER FACE Easy/Medium)

Central issue in detection is tiny faces. They propose a one-stage RetinaNet approach and apply tricks to obtain high performance. Specifically, IoU loss function for regression, two-step classification and regression for detection, utilize max-out operation for classification and use multi-scale testing strategy for inference. 

### SRN (Annotated Faces in the Wild/ Pascal Face) [paper](https://arxiv.org/pdf/1809.02693v1.pdf), [code](https://github.com/ChiCheng123/SRN)

High perfomant detection is challenging when there are many tiny faces. Novel single-shot detector named Selective Refinement Network, introducing novel two-step classification and regression operations selectively into an anchor-based face detector to reduce false positives and improve location accuracy. The SRN consists of two modules: Selective Two-step Classification (STC) module and Selective Two-step Regression module. STC filters out simple negative anchors from low level detection layers to reduce search space for subsequent classifer, while STR adjusts locations/sizes of anchors to provide better initialization for subsequent regressor. What's more they design a Receptive Field Enhancement (RFE) block to provide more diverse receptive fields to capture extreme poses.

### DSFD (FDDB/ PASCAL Face) [paper](https://arxiv.org/pdf/1810.10220v3.pdf), [code](https://github.com/TencentYoutuResearch/FaceDetection-DSFD)

Propose a Feature Enhance Module (FEM) for enhancing original feature maps to extend single shot detector to dual shot detector. Adopt a Progresive Anchor Loss (PAL) computed by two different sets of anchors to faciliate the features. Use an Improved Anchor Matching (IAM) by integrating novel anchor assign strategy into data augmentation to provide better initialization for the regressor. 

### Honourable mentions

- RetinaNet, FaceBoxes, DFace

********************************************************

## Occluded Face Detection

- FAN [paper](https://arxiv.org/pdf/1711.07246v2.pdf), [code](https://github.com/yvette-suyu/Face-Detection-intelligent-city-2018)

********************************************************

## Face Verification

Face verification is the task of comparing a candidate face to another, and verifying whether it is a match. It is a one-to-one mapping: you have to check if this person is the correct one.

### Dual-Agent GANs (IJB-A) [paper](http://papers.nips.cc/paper/6612-dual-agent-gans-for-photorealistic-and-identity-preserving-profile-face-synthesis.pdf) 

Synthesizing realistic profile faces is promising for training pose-invariant models for large-scale unconstrained face recognition by populating samples with extreme poses and avoiding tedious annotation. Learning from synthetic faces may not achieve desired result due to discrepancy between distributions of synthetic and real face images. To narrow this discrepancy, they propose DA-GAN which can improve realism of face simulator's output using unlabeled real faces, while preserving identity information during realism refinement. Dual agents are designed to distinguish real v.s. fake and identities. They use a 3D face model as a simulator to generate profile face images with pose variations. A FCCN as generator to produce high-res images and auto-encoder as discriminator with dual agents. Modify standard GAN (i) pose perception loss; (ii) identity perception loss (iii) adversarial loss with boundary equilibrium regularization term. SOTA on NIST IJB-A unconstrained. 

### ArcFace + MS1MV2 + R100 (LFW/MegaFace) [code](https://paperswithcode.com/paper/arcface-additive-angular-margin-loss-for-deep#code)

[read more](https://github.com/paruliansaragi/Face/blob/master/ArcFace.md)

### SeqFace1 ResNet-64 (Youtube Faces DB) [paper](https://arxiv.org/pdf/1803.06524v2.pdf) [code](https://github.com/huangyangyu/SeqFace#demo)

High-quality datasets are expensive to collect restricting research. SeqFace is proposed to learn discriminative face features. SeqFace can train CNNs by using an additional dataset which includes a large number of face sequences collected from videos. Label smoothing regularization (LSR) and a new proposed discriminative sequence agent (DSA) loss are employed to enhance discrimination power of deep face features via making full use of sequence data. 

### FacePoseNet (IJB-B) [paper](https://arxiv.org/pdf/1708.07517v2.pdf) [code](https://github.com/fengju514/Face-Pose-Net)

Alignment witout explicit facial landmark detection. Landmark detector accuracy is misleading. They find better landmark detection does not indicate better face recognition accuracy on IJB-A/B. Their FPN provides superior 2D and 3D face alignment, and FPN aligns at less computational cost. Faster and more accurate. 

### AIM **NIST #1** (IJB-C) [paper](https://arxiv.org/pdf/1809.00338v2.pdf) [code](https://github.com/ZhaoJ9014/High-Performance-Face-Recognition)

Age-Invariant Model (AIM). Recognizing faces across ages is a big challenge resulting in intra-class variations. 

### Honourable Mentions

- FaceNet, RetinaFace, 

### MobileFaceNets [paper](https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf) [code](https://github.com/deepinsight/insightface) [code](https://github.com/qidiso/mobilefacenet-V2)

Efficient CNN models, MobileFaceNets, less than 1 million parameters tailored for high-accuracy real-time face verification on mobile and embed. Achieves superior accuracy and 2 times speed up over MobileNetV2. After training with ArcFace loss on refined MS-Celeb-1M, the model of 4.0MB size achieves 99.55% accuracy on LFW and 92.59% TAR@FAR1e-6 on MegaFace, comparable to big SOTA CNN models. Inference of 18 ms on mobile.  

********************************************************

## Face Alignment

Face alignment is the task of identifying the geometric structure of faces in digital images, and attempting to obtain a canonical alignment of the face based on translation, scale, and rotation.

### DenseU-Net + Dual Transformer [paper](https://arxiv.org/pdf/1812.01936v1.pdf) [code](https://github.com/deepinsight/insightface) [code inx](https://github.com/deepinx/sdu-face-alignment) 

Current SOTA on images captured in wild revolve around hourglass & U-Nets. They design a novel scale aggregation network structure and channel aggregation. Using deformable convolutions inside stacked dense U-Nets and coherent loss. Able to be spatially invariant. They sow accurate 3D face alignment can assist pose-invariant face recognition achieving SOTA on CFP-FP. 

### DAN (300W) [paper](https://arxiv.org/pdf/1706.01789v2.pdf) [code](https://github.com/MarekKowalski/DeepAlignmentNetwork) [code](https://github.com/justusschock/deep_alignment_network_pytorch) 

Robust face alignment method. DAN has multiple stages, where each stage improves locations of facial landmarks estimated by previous stage. It uses entire face images at all stages, contrary to local patches that are popular. Possible due to landmark heatmaps, which provide visual information about landmark locations estimated at previous stages. Using entire face images allows DAN to handle large variation in head pose and difficult initializations. DAN reduces SOTA failure rate by up to 70%. 

### Joint 3D Face Reconstruction and Dense Face alignment from A single image with 2D-Assisted self-supervised learning  (AFLW2000-3D/AFLW-LFPA) [paper](https://arxiv.org/pdf/1903.09359v1.pdf)[code](https://github.com/XgTu/2DASL)

3D face reconstruction from a single 2D image is challenging with broad applications. Recent methods aim to learn a CNN-based 3D face model that regresses coefficients of 3D Morphable Model (3DMM) from 2D images to render 3D face reconstruction or dense face alignment. The shortage of training data with 3D annotations limits performance of those methods. To mitigate, they propose 2D-assisted self-supervise learning (2DASL) method that can use "in-the-wild" 2D face images with noisy landmark information to improve 3D face model learning. Taking sparse 2D facial landmarks as added info, 2DASL introduces four novel self-supervision schemes that view 2D landmark and 3D landmark prediction as a self-mapping process, including 2D and 3D landmark self-prediction consistency, cycle-consistency over 2D landmark prediction and self-critic over the predicted 3DMM coefficients based on landmark predictions. Mitigating demand for 2D-to-3D annotations without 3D annotations. 

### AWing (COFW) [paper](https://arxiv.org/pdf/1904.07399v1.pdf)

Heatmap regression has become a mainstream approach to localize facial landmarks. CNN & RNN popular in solving comp vision tasks. The loss function for heatmap regression is rarely studied. Propose novel Adaptive Wing Loss that can adapt its shape to different types of ground truth heatmap pixels. Decreasing the loss to 0 on foreground pixels while leaving loss on background pixels. To address fore-back-ground imbalance of pixels, they also propose Weighted Loss Map, which assigns high weights on foreground and difficult background pixels to help training process focus more on pixels that are crucial to landmark localization. To further improve alignment accuracy, they introduce boundary prediction and CoordConv with boundary coordinates. 

### Nonlinear 3D Face Morphable Model (AFLW2000) [paper](https://arxiv.org/pdf/1804.03786v3.pdf)[code](https://github.com/tranluan/Nonlinear_Face_3DMM)

As a classic statistical model of 3D facial shape and texture, 3D Morphable Model (3DMM) is widely used in facial
analysis, e.g., model fitting, image synthesis. Conventional 3DMM is learned from a set of well-controlled 2D face images with associated 3D face scans, and represented by two sets of PCA basis functions. Innovative framework to
learn a nonlinear 3DMM model from a large set of unconstrained face images, without collecting 3D face scans. Specifically, given a face image as input, a network encoder estimates the projection, shape and texture parameters. Two
decoders serve as the nonlinear 3DMM to map from the shape and texture parameters to the 3D shape and texture, respectively. With the projection parameter, 3D shape, and texture, a novel analytically-differentiable rendering layer
is designed to reconstruct the original input face. 

### Dense U-Net (IBUG) [paper](https://arxiv.org/pdf/1812.01936v1.pdf)[code](https://github.com/deepinsight/insightface) [code](https://github.com/deepinx/deep-face-alignment)

## Face Identification

Face identification is the task of matching a given face image to one in an existing database of faces. It is the second part of face recognition (the first part being detection). It is a one-to-many mapping: you have to find an unknown person in a database to find who that person is.

### ArcFace (SOTA MegaFace)

### Deep Residual Equivarient Mapping, DREAM (SOTA IBJ-A) [paper](https://arxiv.org/pdf/1803.00839v1.pdf)[code](https://github.com/penincillin/DREAM)

Many models are poor on profile faces. A key reason is imbalance of training data of frontal vs profile. Moreover, it is hard to learn representation that is geometrically invariant to large pose variations. They hypothesize that there is inherent mapping between frontal and profile faces, and thus a discrepancy in the deep representation space can be bridged by equivariant mapping. DREAM is capable of adaptively adding residuals to the input deep representation to transform profile representation to a canonical pose that simplifies recognition. It improves profile recognition without augmentation, it is light-weight and easily implemented without computational overhead.

### FacePoseNet (IJB-B)

## Ideas

- compacting bounding box captured images into fixed-length vector representations creates a transduction bottleneck? As we've seen in NLP we use transformers etc. to enable us to look back at a long sequence to highlight what is important at time step Tn. Surely in video sequences i.e. real-time face detection we need to use attention mechanisms to encode prediction y at Tn from the attention distribution/source vector.
- Most learn a mapping from image to feature vector. How about GAN feature vector back to representation of input?


