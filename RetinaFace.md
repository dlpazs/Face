# RetinaFace: Single-stage Dense Face Localisation in the Wild

[link](https://arxiv.org/pdf/1905.00641.pdf)

- face localisation remains a challenge in uncontrolled face detection
- They propose a robust single-stage face detector which performs pixel-wise face 
localisation on various scales of faces by taking advantage of joint
extra-supervised and self-supervised multi-task learning
- They contribute by: (1) manually annotating five facial landmarks
on WIDER Face and observe significant improvement in hard face detection
with assistance of this extra supervision signal 
- (2) add a self-supervised mesh decoder branch for predicting pixel-wise 3D shape face information in parallel with existing supervised branches
- (3) SOTA WIDER FACE
- (4) SOTA IJB-C
- (5) light-weight backbone networks, RetinaFace can run in real-time on a single CPU core for VGA-resolution image
- a narrow definition of face localisation may refer to traditional face detection, which aims at estimating face bounding boxes without any scale and position prior
- They have a broader definition which includes face detection, alignment, pixel-wise face parsing and 3D dense correspondence regression
- Inspired by object detection methods
- Differing though, as face detection features smaller ratio variations (1:1 to 1:1.5) but larger scale variations (several to thousand pixels)
- Most recent SOTA focus on single-stage which densely samples face locations and scales on feature pyramids, demonstrating promising performance and yielding faster speed compared to two stage methods
- further improving the single-stage face detection framework and propose a SOTA dense face localisation method by exploiting multi-task losses coming from supervised and self-supervised signals
- Typically, face detection training contains both classification and box regression losses
- Chen et al proposed combined detection and alignment in a joint cascade framework based on the observation that aligned face shapes provide better features for face classification

* The single-stage pixel-wise face localisation employs extra-supervised & self-supervised multi-task learning in parallel with existing box classification and regresion branches
* Each positive anchor outputs (1) face score, (2) a face box, (3) five facial landmarks, (4) dense 3D face vertices projected on the image plane

- Inspired by MTCNN and STN simultaneously detect faces and five facial landmarks
- These have not verified if tiny face detection can benefit from extra supervision of five facial landmarks
- The question they ask is whether they can beat SOTA on WIDER FACE hard set by using extra supervision signal built of five facial landmarks
- Mask R-CNN detection performance is significantly improved by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition and regression
- This confirms that **dense pixel-wise** annotations are beneficial
- for WIDER Face it is not possible to conduct dense face annotation
- since supervised signals cannot by easily obtained
- in Face attention network, an anchor-level attention map is proposed to improve occluded face detection
- the proposed attention map is coarse and does not contain semantic information
- recently self-supervised 3D morphable models have achieved 3D modelling in the wild
- **Mesh decoder** achieves over real-time speed by exploiting graph convolutions on joint shape and texture
- the challenges in getting mesh decoder into single-stage detector are: (1) camera parameters are hard to estimate accurately, (2) joint latent shape and texture representation is predicted from single feature vector (1x1 Conv on feature pyramid) instead of RoI pooled feature which indicates risk of feature shift.
- in the paper they employ a mesh decoder branch through self-supervision learning for predicting a pixel-wise 3D face shape in parallel with the existing supervised branches
- key contributions: single-stage pixel-wise face localisation employing multi-task learning to predict face score, face box, 5 landmarks and 3D position and correspondence of each facial pixel
- SOTA on WIDER FACE beating **Improved selective refinement network for face detection**
- On IJB-C RetinaFace helps improve ArcFace's accuracy showing that better face localisation improves face recognition
- lightweight backbone nets to run real-time on single CPU core
- Extra annotations and code
- **Image pyramid v.s. feature pyramid**: the sliding window paradigm in which a classifier is applied on a dense image can be traced to past decades.
- Viola-Jones did cascading chains to reject false face regions from an image pyramid leading to widespread adoption of scale-invariant face detection
- The emergence of feature pyramid, sliding anchor on multi-scale feature maps dominated the scene
- **Two-stage vs single-stage**: current FD have inherited from object detection in two categories: two stage (Faster R-CNN) single stage (SSD, RetinaNet).
- Two stage employ proposal and refinement enabling high localisation accuracy
- Single stage densely sampled face locations and scales, resulting in unbalanced positive/negative samples during training
- to handle this imbalance, sampling and re-weighting were adopted. Single stage are more efficient and have higher recall rate at the risk of achieving higher false positive rate compromising localisation accuracy
- **Context Modelling** to enhance the model's contextual reasoning power for capturing tiny faces, SSH & pyramid box applied context modules on feature pyramids to enlarge receptive field from Euclidean grids
- To enhance the non-rigid transformation modelling capacit of CNNs, deformable convolution network (DCN) employed novel deformable layer to model geometric transformations
- the winner of WIDER Face 2018 indicates rigid (expansion) and non-rigid (deformation) context modelling are complementary and orthogonal to improve performance of FD
- **Multi-task learning** joint FD and aligned is widely used, aligned faces provide better features for face classification
- In Mask R-CNN FD performance was improved by adding a branch for predicting an object mask in parallel with existing branches. 
- Densepose adopted Mask-RCNN to obtain dense part labels & coordinates within each of the selected regions
- the dense regression brach was trained by supervised learning, the dense branch was a small FCN applied to each RoI to predict pixel-to-pixel dense mapping

## RetinaFace

### Multi-task loss
- for any training anchor $i$, they minimise the following loss: 
- (1) Face classification loss $L_cls (p_i,p_i*)$ where $p_i$ is the predict probability of anchor $i$ being a face and $p_i*$ is 1 for positive anchor and 0 for negative anchor. The classification loss is softmax loss for binary classes (face/not face). 
- (2) Face box regression loss $L_box(t_i, t_i*)$ where $t_i = {t_x, t_y, t_w, t_h}i$ and $t_i* = {t_x*, t_y*, t_w*, t_h*}i$ represent the coordinates of predict box and ground-truth associated box with the positive anchor
- they normalise box regression targers (i.e. centr location, width & height) and use $L_box(t_i,t_i*)=R(t_i-t_i*)$ where $R$ is the robust loss function (smooth $L-1$) defined in Fast r-cnn (Girshick)
- (3) Facial landmark regression loss $L_pts(l_i,l_i*)$ where $l_i={l_x1,l_y1,...,l_x5,l_y5}i$ representing the five facial landmarks and associated ground truth with positive anchor
- similarly, five facial landmark regression employs target normalisation based on anchor centre
- (4) Dense regression loss $L_pixel$
- the loss-balancing parameters increase the significance of better box & landmarks locations from supervision signals

### Dense Regression Branch
- **Mesh decoder** they employ mesh decoder (mesh convolution and mesh up-sampling) which is a graph convolution method based on fast localised spectral filtering
- they use joint shape and texture decoder 







