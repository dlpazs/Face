# Deep Face Recognition: A Survey

[Link](https://arxiv.org/pdf/1804.06655.pdf)

Notes

-  In general, traditional methods attempted to recognize human face by one or two layer representation, such as filtering
responses, histogram of the feature codes, or distribution of the
dictionary atoms.

- What’s worse, most methods aimed to
address one aspect of unconstrained facial changes only, such
as lighting, pose, expression, or disguise. There was no any
integrated technique to address these unconstrained challenges
integrally. 

- Deep learning methods, such as
convolutional neural networks, use a cascade of multiple layers
of processing units for feature extraction and transformation.
They learn multiple levels of representations that correspond to
different levels of abstraction. The levels form a hierarchy of
concepts, showing strong invariance to the face pose, lighting,
and expression changes

- It can be seen
from the figure that the first layer of the deep neural network
is somewhat similar to the Gabor feature found by human
scientists with years of experience. The second layer learned
more complex texture features. The features of the third layer
are more complex, and some simple structures have begun
to appear, such as high-bridged nose and big eyes. In the
fourth, the network output is enough to explain a certain
facial attribute, which can make a special response to some
clear abstract concepts such as smile, roar, and even blue eye.
Deep convolutional neural networks (CNN), the initial layers
automatically learn the features designed for years or even
decades, such as Gabor, SIFT (such as initial layers in Fig.
2), and the later layers further learn higher level abstraction.
Finally, the combination of these higher level abstraction
represents facial identity with unprecedented stability.

![alt text](https://www.researchgate.net/profile/Wang_Mei24/publication/324600003/figure/fig30/AS:651721458081792@1532393912036/Milestones-of-feature-representation-for-FR-The-holistic-approaches-dominated-the-FR.png)

- The hierarchical architecture that stitches together pixels into invariant
face representation. Deep model consists of multiple layers of simulated
neurons that convolute and pool input, during which the receptive-field size of
simulated neurons are continually enlarged to integrate the low-level primary
elements into multifarious facial attributes, finally feeding the data forward
to one or more fully connected layer at the top of the network. The output is
a compressed feature vector that represent the face. Such deep representation
is widely considered the state-of-the-art technique for face recognition.


## A. Background Concepts and Terminology

-First, a face detector is
used to localize faces in images or videos. Second, with the
facial landmark detector, the faces are aligned to normalized
canonical coordinates. Third, the FR module is implemented
with these aligned face images. 

Face recognition can be categorized as face verification and face identification. In either scneario, a tet of known subjects enrolled into the system (the gallery), and during testing, a new subject (probe) is presented.

- Face Verification : computes one-to-one similarity between the gallery and probe to determine if two images are the same. 
- Face Identification : computes one-to-many similarity to determine the specific identity of a probe face. 
- **Closed-set identification** : when the probe appears in the gallery identities, this is referred to as a closed-set identification.
- **Open-set identification** : when the probes include those who are not in the gallery this is open-set.


## B. Components of Face Recognition

-an FR module consists
of face processing, deep feature extraction and face matching,
and it can be described as follows:

$M[F(Pi(Ii)), F(Pj (Ij ))]$

- where Ii and Ij are two face images, respectively; P stands
for face processing to handle intra-personal variations, such as
poses, illuminations, expressions and occlusions; F denotes
feature extraction, which encodes the identity information;
and M means a face matching algorithm used to compute
similarity scores.

1) Face Processing: Ghazi et al. [58] proved that various conditions,
such as poses, illuminations, expressions and occlusions, still
affect the performance of deep FR and that face processing is
beneficial, particularly for poses.
- The face processing methods are categorized as “one-tomany augmentation” and “many-to-one normalization”
- “One-to-many augmentation”: generating many patches
or images of the pose variability from a single image to
enable deep networks to learn pose-invariant representations.
- “Many-to-one normalization”: recovering the canonical
view of face images from one or many images of a
nonfrontal view; then, FR can be performed as if it were
under controlled conditions.

2) Deep Feature Extraction: **Network Architecture.**  The
architectures can be categorized as backbone and assembled
networks
- architectures, such as AlexNet, VGGNet, GoogleNet, ResNet
and SENet [111], [183], [193], [78], [88], are introduced and
widely used as the baseline model
- Moreover, when adopting backbone networks as basic blocks,
FR methods often train assembled networks with multiple
inputs or multiple tasks. One network is for one type of input
or one type of task.
- **Loss Function** : The softmax loss is commonly used as the
supervision signal in object recognition, and it encourages
the separability of features. However, for FR, when intravariations could be larger than inter-differences, the softmax
loss is not sufficiently effective for FR.
• Euclidean-distance-based loss: compressing intravariance and enlarging inter-variance based on Euclidean
distance.
• angular/cosine-margin-based loss: learning discriminative
face features in terms of angular similarity, leading
to potentially larger angular/cosine separability between
learned features.
• softmax loss and its variations: directly using softmax
loss or modifying it to improve performance, e.g., L2
normalization on features or weights as well as noise
injection.

3) Face Matching by Deep Features: After the deep networks are trained with massive data and an appropriate loss
function, each of the test images is passed through the
networks to obtain a deep feature representation. Once the
deep features are extracted, most methods directly calculate
the similarity between two features using cosine distance or
L2 distance; then, the nearest neighbor (NN) and threshold
comparison are used for both identification and verification
tasks. In addition to these, other methods are introduced to
postprocess the deep features and perform the face matching
efficiently and accurately, such as metric learning, sparserepresentation-based classifier (SRC), and so forth

![alt text](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/21117380118ddce47b3c515c5228372c513e61ba/4-Figure3-1.png)
Fig. 3. Deep FR system with face detector and alignment. First, a face detector is used to localize faces. Second, the faces are aligned to normalized canonical
coordinates. Third, the FR module is implemented. In FR module, face anti-spoofing recognizes whether the face is live or spoofed; face processing is used to
handle recognition difficulty before training and testing; different architectures and loss functions are used to extract discriminative deep feature when training;
face matching methods are used to do feature classification when the deep feature of testing data are extracted.

### Different Data Preprocessing Approaches
| Data processing        | Brief Description            | 
| ------------- |:-------------:| 
| one to many       | generating many patches or images of the pose variability from a single image | 
| many to one     |  recovering the canonical view of face images from one or many images of nonfrontal view       |   

## III. NETWORK ARCHITECTURE AND TRAINING LOSS

- As there are billions of human faces in the earth, realworld FR can be regarded as an extremely fine-grained object
classification task.
- For most applications, it is difficult to
include the candidate faces during the training stage, which
makes FR become a “zero-shot” learning task. Fortunately,
since all human faces share a similar shape and texture, the
representation learned from a small proportion of faces can
generalize well to the rest. A straightforward way is to include
as many IDs as possible in the training set

### A. Evolution of Discriminative Loss Functions

- Inheriting from the object classification network such as
AlexNet, the initial Deepface [195] and DeepID [191] adopted
cross-entropy based softmax loss for feature learning. After
that, people realized **that the softmax loss is not sufficient by
itself to learn feature with large margin**, and more researchers
began to explore discriminative loss functions for enhanced
generalization ability.

- Euclidean-distance-based loss played an important role; In
2017, angular/cosine-margin-based loss as well as feature and
weight normalization became popular. It should be noted that,
although some loss functions share similar basic idea, the new
one is usually designed to facilitate the training procedure by
easier parameter or sample selection.

- 1) Euclidean-distance-based Loss : Euclidean-distancebased loss is a metric learning method[230], [216] that embeds
images into Euclidean space and compresses intra-variance
and enlarges inter-variance. The contrastive loss and the triplet
loss are the commonly used loss functions. The contrastive loss
[222], [187], [188], [192], [243] requires face image pairs and
then pulls together positive pairs and pushes apart negative
pairs.

- y. DeepID2 [222] combined the
face identification (softmax) and verification (contrastive loss)
supervisory signals to learn a discriminative representation,
and joint Bayesian (JB) was applied to obtain a robust embedding space

- However, the main problem with the contrastive loss is that
the margin parameters are often difficult to choo

- Contrary to contrastive loss that considers the absolute
distances of the matching pairs and non-matching pairs, triplet
loss considers the relative difference of the distances between
them. . It requires the face triplets, and then it minimizes
the distance between an anchor and a positive sample of the
same identity and maximizes the distance between the anchor
and a negative sample of a different identity. FaceNet uses it. 

- However, the contrastive loss and triplet loss occasionally
encounter training instability due to the selection of effective
training samples

![alt text](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/21117380118ddce47b3c515c5228372c513e61ba/6-Figure5-1.png)
Fig. 5. The development of loss functions. It marks the beginning of deep FR that Deepface [195] and DeepID [191] were introduced in 2014. After
that, Euclidean-distance-based loss always played the important role in loss function, such as contractive loss, triplet loss and center loss. In 2016 and 2017,
L-softmax [126] and A-softmax [125] further promoted the development of the large-margin feature learning. In 2017, feature and weight normalization also
begun to show excellent performance, which leads to the study on variations of softmax. Red, green, blue and yellow rectangles represent deep methods with
softmax, Euclidean-distance-based loss, angular/cosine-margin-based loss and variations of softmax, respectively.

- Center loss [218] and its variant [261], [43], [228]
is a good choice to compresses intra-variance. In [218], the
center loss learned a center for each class and penalized the
distances between the deep features and their corresponding
class centers.

- To handle the long-tailed data, A range loss [261] is used
to minimize k greatest range’s harmonic mean values in one
class and maximize the shortest inter-class distance within one
batch, while Wu et al. [228] proposed a center-invariant loss
that penalizes the difference between each center of classes.
Deng et al. [43] selected the farthest intra-class samples and
the nearest inter-class samples to compute a margin loss.
However, the center loss and its variant suffer from massive
GPU memory consumption on the classification layer, and
prefer balanced and sufficient training data for each identity

- 2) Angular/cosine-margin-based Loss : 
Angular/cosine-marginbased loss. make learned features potentially separable with a larger angular/cosine distance. Liu et al. [126] reformulated the original
softmax loss into a large-margin softmax (L-Softmax) loss

- Due to L-Softmax has difficulty converging, softmax loss
is always combined to facilitate and ensure the convergence,
and the weight is controlled by a dynamic hyper-parameter λ.

- Based on LSoftmax, A-Softmax loss [125] further normalized the weight
W by its L2 norm (kWk = 1) such that the normalized
vector will lie on a hypersphere, and then the discriminative
face features can be learned on a hypersphere manifold with
an angular margin  introduced a
deep hyperspherical convolution network (SphereNet) that
adopts hyperspherical convolution as its basic convolution
operator and that is supervised by angular-margin-based loss.
To overcome the optimization difficulty of L-Softmax and ASoftmax, which incorporate the angular margin in a multiplicative manner, ArcFace [42] and CosineFace [205], AMS
loss [207] respectively introduced an additive angular/cosine
margin cos(θ +m) and cosθ −m. They are extremely easy to
implement without tricky hyper-parameters λ, and are more
clear and able to converge without the softmax supervision.

-  Compared to Euclidean-distancebased loss, angular/cosine-margin-based loss explicitly adds
discriminative constraints on a hypershpere manifold, which
intrinsically matches the prior that human face lies on a
manifold; but Wang et al. [204] showed that angular/cosinemargin-based loss, which used to achieve a better result on a
clean dataset, is vulnerable to noise and becomes worse than
Center loss and Softmax in the high-noise region.

- 3) Softmax Loss and its Variations : In 2017, in addition
to reformulating softmax loss into an angular/cosine-marginbased loss as mentioned above, there are also many works
focusing on modifying it in detail. Normalization of feature
or weight in softmax loss is one of the strategies

- Feature and weight normalization are just effective tricks and should be implemented
with other loss functions.

- the loss functions normalized
the weights only and trained with angular/cosine margin to
make the learned features be discriminative.

-  the loss functions normalized
the weights only and trained with angular/cosine margin to
make the learned features be discriminative.

- Based on the observation of [148] that the L2-norm of features
learned using the softmax loss is informative of the quality of
the face, L2-softmax [157] enforced all the features to have
the same L2-norm by feature normalization such that similar
attention is given to good quality frontal faces and blurry faces
with extreme pose.

- Ring loss [272] encouraged
norm of samples being value R (a learned parameter) rather
8
than explicit enforcing through a hard normalization operation.
Moreover, normalizing both features and weights [206], [130],
[75] has become a common strategy in softmax. In [206],
Wang et al. explained the necessity of this normalization
operation from both analytic and geometric perspectives.

- After
normalizing features and weights, CoCo loss [130] optimized
the cosine distance among data features, and [75] used the von
Mises-Fisher (vMF) mixture model as the theoretical basis to
develop a novel vMF mixture loss and its corresponding vMF
deep features.

### B. Evolution of Network Architecture

- 1) Backbone Network : Mainstream architectures.

### C. Face Matching by deep features

- the cosine distance and L2 are generally employed during testing to measure similarity, then threshold comparison and the nearest neighbour classifier are used to make decision for verification and identification. Here are other common methods:
- 1) Face verification : Metric learning, which aims to find a new metric to make two classes more separable. E.g. JB model does the log of probability of x1,x2 (two faces) belonging to the same identity over the probability of x1, x2 (two faces) belong to different identities. 

- 2) Face identification : after computing cosine, a heuristic voting strategy at the similarity score evel 

- When the distribution of training and test data are the same, the face matching methods are effective. Transfer learning has been introduced for deep FR which utilizes data in relevant source domains (training) to execute in target domain (test)

## IV. Face Processing for Training and Recognition

### A. One-to-Many Augmentation

- can be used to augment not training but the gallery of test data
- **Data augmentation** 
- **3D Model**

## V. Face Databases and Evaluation Protocols

- The development of face databases leads the direction of FR research

### A. Large-scale training data sets and their label noise

- CASIA-Webface : first widely-used public training dataset 0.5M images of 10k celebrities
- MS-Celeb-1M, VGGface2, Megaface
- **Depth v.s. breath** VGGface2 provides large-scale training dataset of depth, which have limited number of subjects but many images. The depth of the dataset enforces the trained model to address a wide range intra-class variations such as lighting, pose and age
- In contrast, MS1M and Megaface offers large-scale training datasets of breadth, which have many subjects but limited images for each subject.
- **Long tail distribution** 
- **Data engineering** 
- **Data bias** exists in most databases. Most datasets are collected from websites and consist of celebrities on formal occasions, smiling, make-up, young. They are largely different from databases captured in real life. Such discrepancies cause a poor performance in applications when directly adopting the pre-trained models. 
- Another serious data bias is the uneven distributions of demographic cohors e.g. race/ethnicity, gender, age. 
- Wang et al. proposed a Racial Faces in the Wild database whose testing can be used to fairly evaluate and compare the recognition ability of the algorithm on different races and training set can enable algorithms to reduce racial bias. [link](http://www.whdeng.cn/RFW/index.html) 
- after baselining commercial APIs and algorithms they show that FR systems work unequally well for different races, the maxmimum difference in error rate between the best and worst groups is 12%. 

### B. Training protocols

- 




