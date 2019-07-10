# Dual Shot Face Detection

[link](https://arxiv.org/pdf/1810.10220.pdf)

* a novel detector that aims to include better feature learning, progressive loss design and anchor assign based data augmentation
* Firstly, they propose a Feature Enhance Module (FEM) for enhancing the original feature maps to extend the SSD to Dual Shot Detector.
* Secondly, they adopt a Progressive Anchor Loss (PAL) computed by two different sets of anchors to effectively facilitate the features.
* Thirdly, they use an Improved Anchor Matching (IAM) by integrating novel anchor assign strategy into data augmentation
to provide better initialization for the regressor
* Previous SOTA can be divided into: Region proposal network (RPN) adopted in Faster-RCNN and employs two stage detection shemes
* RPN is trained end-to-end and generates high-quality region proposals which are further refined by Fast R-CNN
* The other is Single SHot Detector (SSD) based on one-stage methods, which get rid of RPN and directly predict the bounding boxes confidence
* SSD has attracted more attention due to higher inference efficiency
