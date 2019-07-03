# IJB-C

[link](http://biometrics.cse.msu.edu/Publications/Face/Mazeetal_IARPAJanusBenchmarkCFaceDatasetAndProtocol_ICB2018.pdf)

Notes

- IJB-C : advances the goal of robust unconstrained face recognition. Emphasis on occlusion, diversity of occupation, 
geographic origin with the goal of improving representation of global population. IJB-C adds 1,661 new subjects to IJB-B. 
Best benchmark for unconstrained face recognition. 

- Though research has shown vision systems are performing at **near-human levels of face recognition accuracy on constrained
face imagery, the performance of these systems still lags behind human performance on unconstrained imagery**

- Because many practical applications of face recognition, e.g. surveillance, necessarily operate on unconstrained
imagery, it is critical to improve unconstrained face recognition performance. A practical unconstrained face recognition system must successfully perform face detection, verification, and identification regardless of subject conditions
(pose, expression, occlusion) or acquisition conditions (illumination, standoff, etc.).

|Dataset |# subjects| # images| avg. # img/subj| # videos| avg. # vid/subj| pose variation|
|--------|--------|--------|--------|--------|--------|--------|
|IJB-C| 3,531| 31,334| 6| 11,779| 3| full|
|IJB-B[17]| 1,845| 21,798| 6| 7,011| 4| full|
|IJB-A [9]| 500| 5,712| 11| 2,085| 4| full|
|LFW [7]| 5,749| 13,233| 2| 0| N/A| limited|
|YTF [18]| 1,595| 0| N/A| 3,425| 2| limited|
|PubFig [10]| 200| 58,797| 294| 0| N/A| limited|
|VGG [13]| 2,622| 982,803| 375| 0| N/A| limited|
|MegaFace [8]| N/A| 1M| N/A| 0| N/A| full|
|MF2[12]| 672,057| 4.7M| 7| 0| N/A| limited|
|WIDER FACE [19]| N/A| 32,203| N/A| 0| N/A| full|
|CASIA Webface [20]| 10,575| 494,414| 47| 0| N/A| limited|
|UMDFaces [2]| 8,277| 367,888| 44| 22,075| 31| full|

- One of the key limitations of the above datasets is that
commodity face detectors such as Viola-Jones (V-J) [16]
were used to collect the faces in the dataset. The V-J face
detector was not designed to detect faces with significant
degrees of roll, pitch, or yaw, so using such a detector to
construct a dataset excludes truly unconstrained imagery
from it, reducing the dataset’s relevance in solving the unconstrained face recognition problem

- MegaFace includes one million faces and is
intended to be used only as a distractor set, whereas MF2
has 672K unique identities but is intended to be used only
as a training set. 

- WIDER FACE, a large scale face detection dataset released in 2016, made significant strides towards addressing
the data quantity problem associated with evaluation of face
detection algorithms [19]. However, utility of this dataset
is limited to advancing face detection only, since subject
identity labels are not provided.

- (IJB-A) dataset in 2015 marked a
milestone in unconstrained face recognition research [6][9].
When released, results from multiple submissions to the
challenge showed significantly worse recognition performance compared to the previously mentioned datasets. As
of 2017, performance on IJB-A is approaching saturation,
with a top true accept rate of 96.1% at a 1.0% false accept
rate

- IJB-C : All subjects in the dataset are ensured to appear in at least
two still images and one video. The bounding boxes and
metadata labels were all labeled using the crowdsourcing
platform Amazon Mechanical Turk (AMT).
