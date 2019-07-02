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

-  FR can be categorized as face verification and
face identification. In either scenario, a set of known subjects is
initially enrolled in the system (the gallery), and during testing,
a new subject (the probe) is presented. Face verification
computes one-to-one similarity between the gallery and probe
to determine whether the two images are of the same subject,
whereas face identification computes one-to-many similarity
to determine the specific identity of a probe face. When the
probe appears in the gallery identities, this is referred to as
**closed-set identification**; when the probes include those who
are not in the gallery, this is **open-set identification**

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



