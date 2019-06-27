# Dlib Model Architecture

- https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py
- https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-can-i-re-train-the-face-encoding-model-to-make-it-more-accurate-for-my-images

# ArcFace

- in many facial recognition models, the spread of features produced by its loss function is large causing boundaries between classes to be blurred and making it hard for the model to correctly classify images.
- Arcface's Additive Angular Margin helps reduce this spread by pushing images of same individual whilst pushing away images of different individuals
- a major challenge in model creation for large scale image recognition is finding a loss function that can enhance discriminative power between clusters
- the authors propose a new loss called ArcFace:
	- it utilises 'Additive Angular Margin Loss' to reduce intra-class distance whilst increasing inter-class distance of features
	- outperforms Softmax/Triplet/Centre loss

## Some common loss functions: Softmax

- works by inputting vector of digits and normalizing them to be a probability distribution for each class 
- the loss function then calculates the distance between what the distribution of the output should be and what the original distribution really is
- softmax does not enforce separation between classes which causes the classes to be closely clustered together
- Softmax loss does not explicitly optimise the feature embedding to enforce higher similarity for intraclass samples and diversity of inter-class samples, which results in a performance gap for deep face recognition under large intra-class appearance variations (e.g. pose variations/age gaps) and large scale test scenarios (e.g. million or trillion pairs). 

## Centre Loss

- Centre loss tries increase disparity of classes produced by softmax by calculating the centre for each class and moving its intra-correlated features closer towards thus creating class compactness
- Firstly, it improves softmax but calculating the centre for each class is is computationally expensive since it has to calculate the distance of all features to calculate the centre for each class
- Secondly, the centre chosen is not always totally accurate because all features cannot be calculated ahead of time so centres need to be created and redefined in each batch
- Finally, the distance penalty that's applied to features is calculated using Euclidean distance which isn't the best way to separate softmax losses

- Simultaneously learns a centre for deep features of class and penalises the distances between the deep features and their corresponsing class centres in the Euclidean space to achieve intra-class compactness and their inter-class disparity
- **Drawbacks of Centre loss** 
- 1. updating the actual centres during training is extremely difficult as the number of face classes available for training has recently dramatically increased

## Triplet Loss

- Called "triplet" cause you compare 3 images:
	- Anchor image (a)
	- Positive image (same subject as the anchor) (p)
	- Negative image (different subject to the image) (n)

- the goal is to make the distance between the 2 images of the same person be shorter than the anchor and imposter
- this is done by adding a margin value to the positive image

- **Drawbacks of Triplet loss** 
- 1. There's a 'combinatorial explosion' in the number of face triplets, especially for large datasets, leading to a significant increase in the number of iteration steps
- as larger number of images leads to exponential number of pairings

- 2. Semi-hard sample mining (where the positive and negative images have a similar distance) is a difficult task for effective model training 
- Semi-hard sampling required to train effectively, if the function were to compare two random images of person x and y, the chances are the distance between the two x images would be much smaller than the distance to y which would satisfy the loss but wouldn't teach the model much 
- instead it is much more effective to compare two similar images such as x and x-very similar as the model would have to work hard to adjust its weights and have a better classification
- this process of semi-hard sampling is very computationally expensive

## SphereFace Loss

- Most losses use Euclidean-based margin to optimise separability between features
- Sphere based researchers found that Euclidean measures are not suitable for softmax which has a naturally angular distribution, SphereFace utilises softmaxes natural angular distribution by imposing discriminative constraints on a hypersphere manifold allowing inter and intra-loss values to be controlled by a parameter $m$
- this method is called angular softmax or a-softmax 
- by constraining the weights and biases, the new decision boundary only depends on $beta_1$ and $beta_2$ so it's then just matter of adding an integer $m$ to control the decision boundary
- Features learned by the original softmax loss have an intrinsic angular distribution, so directly combining Euclidea margin constraints with softmax loss is not reasonable
- SphereFace takes advantage of this angular distribution by applying an angular margin to separate the features, a method called A-softmax
- the decision boundary in softmax loss is $(W_1 - W_2)x + b_1 - b_2 = 0$ where $W_i$ and $b_i$ are weights and biases in softmax loss
- If we define $x$ as a feature vector and constrain $||W_1|| = ||W_2|| = 1$ and $b_1 = b_2 = 0$, the decision boundary becomes $||x||(cos(theta_1)-cos(theta_2))=0$ where $theta_i$ is the angle between $W_i$ and $x$. 
- the new decision boundary only depends on $theta_1$ and $theta_2$ 

- $m$ controls the size of the angular margin, simultaneously enlarging the inter-class margin and compressing the intra-class angular distribution
- $m$ is computed as an integer as a value of $\geq{1}$, if $m = 1$ then the decision planes of category 1 and category 2 are on the same plane  
- if $\geq 2$ then there are 2 decision planes for category 1 and 2, that indicates the maximum angle with the classication is smaller than the small of the angle of other classes by $m$ times
- A-softmax approximates the optimal value of $m$, with it's criteria being that the maximal intra-class distance should be smaller than the minimal intra-class distance
- there are 2 main drawbacks: 
	- the integer value of $m$ which causes the curve of the target logit i.e. the logit that corresponds to the ground truth label to be very steep thus hinder convergence
	- the decision margin for A-softmax depends on $\theta$ which leads to different margins for different classes, as a result, in the decision space some inter-class features have larger margin while others have a smaller margin which reduces its discriminating power

- Now it can simply introduce an integer $m(m \geq{1}$ to control the decision boundary
- in a binary-class example, the decision boundaries for class 1 and class 2 become $||x||(cos(m\theta_1)-cos(\theta_2)) = 0$ and $||x||(cos(\theta_1) - cos(m\theta_2))=0$, respectively
- **Drawbacks of SphereFace Loss**
- 1. The decision of margin of A-softmax depends on $theta$, which leads to different margins for different classes
- 2. Empirically, the softmax loss dominates the training process, because the integer-based multiplicative angular margin makes the target logit curve very precipitous and thus hinders convergence

## CosFace Loss

- Uses a large Margin Cosine Loss (LMCL) to maximize inter-class variance and minimise intra-class variance and overcome shortcomings of SphereFace
- it does this by defining the decision margin in cosine space, which SphereFace's A-softmax which defines it in angular space, it starts by reformulating the softmax loss by L2 normalizing the features and weights to remove radial variants this addresses A-softmax's first issue of producing different margins for different classes dependent which is the result of depending on the value of $\theta$
- as with A-softmax a margin value is added to increase the inter-class distance and reduce intra-class distance 
- CosFace's loss function maximizes cos theta 1 and minimizes cos theta 2 for C1 to perform the large margin classification
- this is superior to A-softmax decision boundary who's margin is not consistent for all theta values making the decision boundaries hard to optimize

## Additive Angular Margin Loss

- ArcFace further improves discriminative power by adding additive angular margin 
- unlike CosFace which applies an angular margin directly to the target logit, ArcFace applies the inverse of the angle using the arc-cos function before using the cosine function to get back the target logit, then it re-scales the logit by a fixed feature norm and the rest is the same as the softmax loss function
- ArcFace uses an angular margin similar to SphereFace and CosFace to maximize face class separability
- it utilises the arc-cosine function to calculate the angle between the current feature and the target weight
- afterwards, it adds an additive angular margin to the target angle and we get the target logit back again by cosine function 
- then it re-scales all logits by a fixed feature norm, and the subsequent steps are exactly the same as in the softmax loss


- start with a normal softmax loss function, which doesn't optimise feature embedding to maximise face class separability
- the bias is fixed to 0
- the logit is transformed as $W^{T}_{j} x_i = ||W_j||||x_i|| cos \theta_j$, where $\theta_j$ is the angle between weight $W_j$ and the feature $x_i$
- the individual weight is fixed to ||W_j||=1$ by $l_2$, as well as the embedding feature which is fixed to $||x_i||$ by $l_2$ normalisation and re-scaled to $s$.
- the normalisation step on features and weights makes the predictions only depend on the angle between the feature and the weight

- the embedding features are distributed around each feature centre on the hypersphere, we add an additive angular margin penalty $m$ between $x_i$ and $W_{y_i} to simultaneously enhance intra-class compactness and inter-class discrepancy
- Since the proposed additive angular margin penalty is equal to the geodesic distance margin penalty in the normalised hypersphere, the method is called ArcFace

# ArcFace the paper

- One of the key challenges in feature learning using DCNN's for large scale face recognition is designing the appropriate loss function that enhances discriminative power
- Centre loss penalises distance between features and their class centres in Euclidean space to achieve intra-class compactness
- SphereFace assumes the linear transformation matrix in the last fully connected layer can be used as a representation of the class centres in an angular space and penalises angles between deep features and their corresponding weights in a multiplicative way
- more recently it has been popular to use margins in well-established loss functions to maximize face class separability
- This paper proposes Additive Angular Margin Loss (ArcFace) to obtain highly discriminative features 
- has a clear geometric interpretation due to the exact correspondence to the geodesic distance
***
- **NB**: geodesic distance : the distance between two vertices in a graph is the number of edges in a shortest path also called graph geodesic connecting them. This is also know as geodesic distance. A geodesic line is thus the shortest path between two points on a curved surface. They are the analogue of a straight line on a plan surface or whose sectioning plane at all points along the line remains normal to the surface. It is a way of showing distance on an ellipsoid whilst that distance is being projected onto a flat surface. ![alt text](https://qph.fs.quoracdn.net/main-qimg-5b21d634622cc41141207b8325f13843) The image shows a planar distance in orange and a geodesic distance of that planar distance in blue. So it can be thought of as a generalization of a straight line to curved surfaces. 
***
- based on centre and feature normalisation, all identities are distributed on a hypersphere
- to enhance intra-class compactness and inter-class discrepancy, they consider four kinds of Geodesic distance constraints; (A) Margin-Loss : insert a geodesic distance margin between the sample and centres. (B) Intra-loss : decrease geodesic distance between the sample and corresponding centre. (C) Inter-loss : increase geodesic distance between different centres. (D) Triplet-loss : insert a geodesic distance margin between triplet samples. 
- The proposed Additive Angular Margin Loss corresponds to the geodesic distance margin penalty (A), to enhance the discriminative power of the face recognition model. 
- Face representation using DCNN's embed a representation after pose normalisation of a face image, into a feature vector that has small intra-class and larger inter-class distance
- There are two main lines to train DCNN's for face recognition
- 1) Those that train multi-class classifier which separate identities in the training set using softmax classifier
- 2) Those that learn directly an embedding such as triplet loss
- Drawbacks of softmax loss and triplet loss : 
- For softmax, (1) the size of linear transformation matrix increases linearly with identifies; (2) the learned features are separable for the closed-set classification problem but not discriminative enough for open-set face recognition problem
- For triplet loss, (1) there is the combinatorial explosion in the number of face triplets for large data-sets leading to increase in interation steps; (2) semi-hard sampling is difficult problem during training
- Centre loss enhances softmax loss, the Euclidean distance between each feature vector and its class centre, to obtain intra-class compactness. Updating the centres during training is extremely difficult as the number of face classes available for training has drastically increased
- by observing the weights from the last fully connected layer of a classification DCNN trained on softmax bear conceptual similarities with the centres of each face class, a multiplicative angular margin penalty enforce extra intra-class compactness and inter-class discrepancy simultaneously, leading to better discrimative power
- SphereFace introduced the important idea of angular margin, their loss requires a series of approximations to be computed which was unstable
- to stabilise training, they propose a hybrid loss which includes softmax. 
- The softmax dominates the training process, because the integer-based multiplicative angular margin makes the target logit curve very precipitous and hinders convergence
- CosFace adds cosine margin penalty to the target logit, which improves performance and removes the need for joint supervision from the softmax loss
- This paper proposes ArcFace to further improve discriminative power
- the dot product between the DCNN feature and last fully connected layer is equal to the cosine distance after feature and weight normalisation
- they utilise the arc-cosine function to calculate the angle between the current feature and the target weight
- then they add an additive angular margin to the target angle, and get the target logit back again by the cosine function
- then they re-scale all logits by a fixed feature norm and the subsequent steps are exactly the same as in the softmax loss
- **Engaging**. ArcFace optimises the geodesic distance margin by virtue of the exact correspondence between the angle and arc in the normalisd hypersphere. 
- Achieves SOTA
- **Easy.** Only needs a few lines of code and easy to implement in Pytorch/TF/MxNet. It converge well and doesn't need other loss functions
- **Efficient**. Adds only negligible computational complexity at training
- Approach : bias is set to 0. The logit i.e. weights transpose dotted with x, is transformed into $||W_j||||x_i||cos\theta_j$, where $\theta_j$ is the angle between weight $W_j$ and the feature $x_i$
- they then fix the individual weight $||W_j|| = 1$ by $l_2$ normalisation
- they also fix the embedding feature $||x_i||$ by $l_2$ normalisation and re-scale it to $s$.
- the normalisation of features and weights makes predictions only depend on the angle between the feature and the weight
- the learned embedding features are thus distributed on a hypersphere with a radius of $s$. 
- as embedding features are distributed around each feature centre on the hypersphere, we add an additive angular margin pentaly $m$ between $x_i$ and $W_{y}{i}$ to simultaneously enhance the intra-class compactness and inter-class discrepancy
- since the proposed additive angular margin penalty is equal to the geodesic distance margin penalty in the normalised hypersphere they name their method ArcFace


![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRP5jO_0qhIVeLu47xzFKXPyNZamQhsP7D6FtIBm5OjDoNwTLqxJQ)

![alt text](https://4.bp.blogspot.com/-vPyIn-UXH1I/WxaaNfZ0FWI/AAAAAAAAwkI/fjxi9IVfImoy81sth-4FHYMv2DZ0qCasgCEwYBhgL/s1600/1.PNG)

- the softmax loss provides roughly separable feature embedding but produces noticeable ambiguity in decision boundaries, while the proposed ArcFace loss can enforce more evident gap between the nearest classes
- Based on feature normalisation, all face features are pushed to the arc space with a fixed radius $s$. The geodesic distance gap between closest classes becomes evident as the additive angular margin penalty is incorporated.

### Comparison with SphereFace and CosFace

- in all 3, 3 different kinds of margin are proposed e.g. multiplicative angular margin $m_1$, additive angular margin $m_2$ and additive cosine margin $m_3$
- from a numerical analysis point of view, different margin penalties, no matter add on the angle or cosine space, all enforce intra-class compactness and inter-class diversity by penalising the target logit
- the additive angular margin has a better geometric attribute as the angular margin has the exact correspondence to the geodesic distance
- They compare decision boundaries. ArcFace has a constant linear angular margin unlike SphereFace and CosFace only have a nonlinear angular margin. 
- **Intra-loss** is designed to improve intra-class compactness by decreasing the angle/arc between the sample and ground truth centre
- **Inter-loss* tries to enhance inter-class discrepancy by increasing the angle/arc between different centres
- Triplet-loss tries to enlarge the angle/arc margin between triplet samples

### Experiementation & Implementation Details

- first to employ ethnic specific annotators
- for data preprocessing they generate normalised face crops 112x112 by utilising five facial points/landmarks
- for the embedding network they use CNN architectures ResNet50 and ResNet100, Batc Norm after last conv layer, BN-Dropout-FC-BN structure to get final 512-D embedding
- set feature scale $s$ to 64 and angular margin $m$ at 0.5







 
