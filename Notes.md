# Testing of Face Recognition Models

[Balancing Facial Recognition Performance and Accuracy in the Real World](https://safr.com/general/balancing-facial-recognition-performance-and-accuracy-in-the-real-world/)

- NIST conducts an on-going battery of tests, known as the Face Recognition Vendor Test (FRVT), to measure the key characteristics of facial recognition algorithms, including accuracy, performance, and bias.
- NIST not only measures specific characteristics of facial algorithms, such as performance, accuracy, and bias, the standards and measurements body also reports on those attributes by image type, such as visa photos, mugshots, webcam, or “wild” images. Wild images are camera unaware faces captured on video: complex because the faces may be tilted, with wide yaw and pitch pose variation. Not to mention there may be many faces within a video frame. Wild images are challenging facial qualities that are precisely the type of real-world conditions for which SAFR was designed. NIST conducts its tests of facial recognition using still photographs. Facial recognition in live video requires concerted optimization in acquisition, accuracy, and speed.
- SAFR from RealNetworks is the most accurate high-performance facial recognition algorithm for live video as tested by NIST.

[Nist November](https://www.nist.gov/sites/default/files/documents/2019/01/29/frvt_report_2019_01_29.pdf)


## [A Face Recognition System Based on Eigenfaces Method](https://reader.elsevier.com/reader/sd/pii/S2212017312000242?token=55E4E396A177255FFC34220CFBCA370170540FC2334CF574CECFF224B4EF771171811F6D28D1019EF7B4A9309DCBC3F8)

## [A Performance Comparison of Loss Functions for Deep Face Recognition](https://arxiv.org/pdf/1901.05903.pdf)

## [Face recognition accuracy of forensic examiners, superrecognizers, and face recognition algorithms](https://www.pnas.org/content/pnas/115/24/6171.full.pdf)

***
# Post-processing

Improving precision and recall of face recognition in SIPP with combination of modified mean search and LSH
https://arxiv.org/pdf/1709.03872.pdf

Single Image Per Person (SIPP), searching over complicated database, not been solved.
1) Modified SVD based augmentation to increase intra-class variations (even for person with only one image)
2) Combination of modified mean search and LSH to help get the most similar personID in a complicated dataset. 
3) No need to retrain of DNN model and easy to extend.
“We do some practical testing in competition of Msceleb challenge-2 2017 which was hold by Microsoft Research, great improvement of coverage from 13.39% to 19.25%, 29.94%, 42.11%, 47.52% at precision 99%(P99) would be shown latter, coverage reach 94.2% and 100% at precision 97%(P97) and 95%(P95) respectively. As far as we known, this is the only paper who do not fine-tuning on competition dataset and ranked top-10. A similar test on CASIA WebFace dataset also demonstrated the same improvements on both precision and recall. ”

“Baseset, there are 20,000 persons in the baseset. Each person has 50-100 images for training, and about 5 images for testing. Novelset, there are 1,000 persons in the novelset. Each person has only 1 image for training, and 20 images for testing.”

- Test set they use novelset persons but search for person IDs in both Baseset and Novelset. 

Modified SVD based face augmentation

Decompose face into 2 complementary parts: 
- First part constructed by SVD basis images associated with several largest singular values.
- Second part is constructed by the other low-energy basis images.
- The first part preserves most of the energy of an image and reflects the general appearance of the image (so not much change to the original image using largest singular values). 
- The second part is the difference between the original image and the first part, it reflects the variations of the same class face images such as pose and expressions

- Given post-alignment face image, we want the sum over each singular value of A, U and V eigenvectors of AA Transpose. 
- Reserve 95%, 90%, 85% which is the balance between degree of variations and image quality. 
- Faces i,j,k where i,j,k = 1.0,0.95, 0.9, 0.85 = merge(Ri, Gj, Bk).  
- They do SVD augmentation on each of RGB channel and merge them to obtain more varaitions, one face image becomes 4*4*4=64 in total 3 channels and each has 4 energy choice. With the different percentages of energy reserving for each channel, more variations would get compared with formal methods. We expand feature vector point of a specialized face to a sphere surrounding the point like Figure 3. Note because of different pose and expression, the surrounding sphere may not be in the middle of the feature space of a particular person. 

- “First, a Dlib3 based face detection, face alignment and face feature extractor method were used in whole of the evaluation. Dlib deep neural network face recognition tools has a 99.38% accuracy on the standard LFW face recognition benchmark, which is comparable to other state-of-the-art methods for face recognition until early of 2017. It’s essentially a version of the ResNet34 network from the paper Deep Residual Learning for Image Recognition by He, Zhang, Ren, and Sun with a few layers removed and the number of filters per layer reduced by half.”

- “we just do a brute force search over all face feature vectors of 21,000 person in baseset and novelset(SVD degrade features are not included), search for the nearest face in totally 1,169,166 face images. And then we search for the most similarity personID for test set in way of compare with all feature vectors, similarity score defined below, and maximum similarity score return the particular personID. After a truly time consuming search, the final result is 13.39%@P99, 33.41%@P97, 56.87%@P95.”

- SimilarityScore = 1.0/ (1.0 + dist(X, Y))
- Dist(X,Y) = sqrt ( Sum (xi- yi)^2 ) 

- “Then, when SVD degrade features were included, and the result is 19.25%@P99, 39.64%@P97, 72.69%@P95, which emphasize that SVD is truly useful for SIPP face recognition. Theoretically, when SVD face feature vectors included, it seems like that we added some intra-class variations for person in novelset, or more precisely, we have expanded high dimension feature space for novelset face image from one point to a hypersphere around the point. Showing in figure 3, in the right circle, expanded from the blue point to the blue circle. It’s not a good results and we began to optimize the search methods…”

- “when only 2 methods had added, that is to say, when a precision of 95% ensured, we can recognize all the people in the test set.”

Methods for Searching the personID

The focus is not on training a NN to min intra-class and max inter-class but focus on a search strategy for finding the corresponding personID. The emphasis is on the combination of modified mean search and LSH methods.

Brute-force search with SVD degraded face

SVD face with 95%, 90%, 85% energy reserved were included in the search space. 
On the one hand, it’s a way to add intra-class variations for people in the test (novel) set. 
On the other, SVD added faces is also a way for us to expand the representation of a person from one point to one hyperspace surrounding the point, although the hyperspace is not in the middle of the person feature space. 
They find that, it’s easy for us to find a person whose expression, light-conditions, POV and so on is quite similar to the search face but comes from another personID. This can be explained by figure 3, although the hyperspace of the novelset person has been expanded with SVD faces, but the expansion is limited because SVD faces can only add variations in one (several) way(s), or in other words, the expanded hyperspace is limited. How about we just utilize the mean feature vector to represent a person.

Mean search with SVD degraded face

Brute-force is time-consuming, so they move to mean search but the issue they found was that faces in novelset had only one image, and its mean vector is itself (the mean adds no extra info)
So for the novelset the mean vector was computed over all SVD degraded feature vectors, this means the main info of a person should be utilized in order to not be misguided by variations such as expressions/lighting etc.
One issue: the mean feature vector is not in the middle of the high dimensional hyperspace of the person, but located on the one and only face feature vector we have. As that one feature vector is a little far from the middle of the hyperspace of the person, it will always lead to low similarity scores despite it being within the same person.
Does brute force search not work at all? They say no, but it can help get more distinguishable similarity scores when two faces belong to one person so they combine mean and brute force search. 

Combination of modified mean search and brute-force search

Mean search is a way for us to ensure a minimum coverage and precision, and brute force is a good way for us to get more distinguishable similarity scores which increases coverage a step further. 
In table 2: if we can assign a more distinct similarity score when two faces came from the same person with a higher confidence, then coverage of a particular precision can be improved. If we want higher similarity scores, brute force can help as seen with d2 in figure 3. The same is true for figure 5, when we do mean search, d3 would not be chosen because of misclassification. In figure 5 we see that d3 would be chosen but this is wrong, if we do the mean vectors of d1 and d2 we see that d2 is closer than d1 thus is correctly classified.
They find that a feature vector with a higher similarity score which exceeded the similarity score of the mean feature vector with a threshold T, that would be an indication of that being the same person, if not, they always come from two persons. Suppose after mean search, we find that the most similar score s1 corresponding to id1, and after brute force search, another most similar score s2 and that correspond to id2. The equation is in (7). 
If mean and brute search find the same person ID then get the max score between mean and brute search because we have a high confidence of the correct id. If mean and brute search indicate different ids, choose the one who’s similarity score exceeds the other over some threshold T. If below this threshold, then choose the minimum score between s1 and s2, (T set to 0.03) of id1, if the mean search id for mean search has a higher confidence than brute force search.

Combination of modified mean search and LSH

Brute force is slow and cannot be used in real-time. If noise exists in a feature vector, brute force tends to mis-match. In order to mitigate this, LSH-based Nearest Neighbour search was used to do the combination with mean search.
Each method used makes sense intuitively:
SVD degraded faces are used to expand a single image from one point to a hypersphere
Mean search is used to compute similarity using fundamental information of a face ignoring improving pose/expression invariance
Combinations of brute-force and mean search create higher confidence scores of correct searches
Combining mean search and LSH helps ignore noise in the database

***
# Post-processing continued

https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23

Very quick!

Audio fingerprinting is the process of identifying unique characteristics from a fixed duration audio stream. Such unique characteristics can be identified for all existing songs and stored in a database. When we hear a new song, we can extract similar characteristics from the recorded audio and compare against the database to identify the song. However, in practice there will be two challenges with this approach:
1) High dimensionality of the unique characteristic/feature vector required to identify songs
2) Comparison of the recorded audio features against features of all songs in the database is expensive in terms of time and memory
The first challenge can be addressed using a dimensionality reduction technique like PCA and the second using a combination of clustering and nearest neighbor search. Locality Sensitive Hashing (hereon referred to as LSH) can address both the challenges by
1) reducing the high dimensional features to smaller dimensions while preserving the differentiability
2) grouping similar objects (songs in this case) into same buckets with high probability

LSH is a hashing based algorithm to identify approximate nearest neighbors. In the normal nearest neighbor problem, there are a bunch of points (let’s refer to these as training set) in space and given a new point, objective is to identify the point in training set closest to the given point. Complexity of such process is linear [for those familiar with Big-O notation, O(N), where N is the size of training set]. An approximate nearest neighboring algorithm tries to reduce this complexity to sub-linear (less than linear but can be anything). Sub-linear complexity is achieved by reducing the number of comparisons needed to find similar items.

LSH works on the principle that if there are two points in feature space closer to each other, they are very likely to have same hash (reduced representation of data). LSH primarily differs from conventional hashing (aka cryptographic) in the sense that cryptographic hashing tries to avoid collisions but LSH aims to maximize collisions for similar points. In cryptographic hashing a minor perturbation to the input can alter the hash significantly but in LSH, slight distortions would be ignored so that the main content can be identified easily. The hash collisions make it possible for similar items to have a high probability of having the same hash value.

Locality Sensitive Hashing (LSH) is a generic hashing technique that aims, as the name suggests, to preserve the local relations of the data while significantly reducing the dimensionality of the dataset.

Now that we have established LSH is a hashing function that aims to maximize collisions for similar items, let’s formalize the definition:

A hash function h is Locality Sensitive if for given two points a, b in a high dimensional feature space, 1. Pr(h(a) == h(b)) is high if a and b are near 2. Pr(h(a) == h(b)) is low if a and b are far 3. Time complexity to identify close objects is sub-linear.

Detour — Random Projection Method
Random projection is a technique for representing high-dimensional data in low-dimensional feature space (dimensionality reduction). It gained traction for its ability to approximately preserve relations (pairwise distance or cosine similarity) in low-dimensional space while being computationally less expensive.
The core idea behind random projection is that if points in a vector space are of sufficiently high dimension, then they may be projected into a suitable lower-dimensional space in a way which approximately preserves the distances between the points.

Consider a high-dimensional data represented as a matrix D, with nobservations (columns of matrix) and d features (rows of the matrix). It can be projected onto a lower dimensional space with k-dimensions, where k<<d, using a random projection matrix R. Mathematically, the lower dimensional representation P can be obtained as 

[ Projected(P) ] k by n = [ Random (R ) ] k by d [ Original (D) ] d by n

Columns of the random projection matrix R are called random vectors and the elements of these random vectors are drawn independently from gaussian distribution (zero mean, unit variance).

In this LSH implementation, we construct a table of all possible bins where each bin is made up of similar items. Each bin will be represented by a bitwise hash value, which is a number made up of a sequence of 1’s and 0’s (Ex: 110110, 111001). In this representation, two observations with same bitwise hash values are more likely to be similar than those with different hashes. Basic algorithm to generate a bitwise hash table is:

1) Create k random vectors of length d each, where k is the size of bitwise hash value and d is the dimension of the feature vector.
2) For each random vector, compute the dot product of the random vector and the observation. If the result of the dot product is positive, assign the bit value as 1 else 0
3) Concatenate all the bit values computed for k dot products
4) Repeat the above two steps for all observations to compute hash values for all observations
5) Group observations with same hash values together to create a LSH table


https://colab.research.google.com/drive/1RCR89KrTv6vu93LePGeU8d7QxKIuWJEm#scrollTo=BfxCAUIfOC0r


We can infer that vec1 and vec2 are more likely to be similar (same hash value) than vec1 and vec3 or vec2 and vec3. We can observe that the cosine similarity is maximum for vec1 and vec2 compared to other two combinations, which confirms the output of random projection method.

The intuition being that if two points are aligned completely, i.e. have perfect correlation from origin, they will be in the same hash bin. Two points separated by 180 degrees will be in different bins and two points 90 degrees apart have 50% probability to be in same bins. Because of the randomness it is not likely that all similar items are grouped correctly. To overcome this limitation a common practice is to create multiple hash tables and consider an observation a to be similar to b, if they are in the same bin in at least one of the tables. Multiple tables generalize the high dimensional space better and amortize the contribution of bad random vectors. The number of hash tables and size of the hash value (k) are tuned to adjust the trade-off between recall and precision. 


1) Construct a feature vector for all faces in DB
2) Construct LSH Hash tables using the above defined classes with appropriate choice for number of tables and hash size.
3) For a newly recorded audio, construct the feature vector and query the LSH tables
4) Compare the feature vector of the recorded audio with the matches returned in step 3. Metrics for comparison can be L2 distance, cosine similarity or Jaccard similarity, depending on the elements of feature vector.
5) Return the result that has lowest/highest metric value (depending on the chosen metric) as the match


https://github.com/pixelogik/NearPy

***
# Some Papers

ArcFace:
https://arxiv.org/pdf/1801.07698.pdf 

They try to find the best loss function in face recognition and find that the Additive Angular Margin Loss (ArcFace) is the best at enhancing the discriminative power of face recognition. They use a Deep CNN (DCNN) to embed a face and map the face (after normalizing for the pose) into a feature that has a small intra-class and large inter-class geodesic distance . The two main approaches for DCNN in face recognition (FR) is to use multi-class classification to separate identities (using softmax classifier) or by learning directly an embedding of a face. Both methods work. Variants on the softmax loss include: centre loss (Euclidean distance between each feature vector and its class centre, intra-class compactness and inter-class dispersion is done via joint penalisation of the softmax loss). They employ Additive Angular Margin Loss to improve discriminative power and stabilise training. The dot product of the DCNN feature and last fully connected layer is equal to the cosine distance after feature and weight normalisation.  In the mathematical field of graph theory, the distance between two vertices in a graph is the number of edges in a shortest path (also called a graph geodesic) connecting them. This is also known as the geodesic distance.

FaceNet:
SUMMARY:
FaceNet directly learns a mapping from face images to a compact Euclidean space where distances correspond to face similarity. Squared L2 distances in embedding space correspond to face similarity (faces of the same person have small distances and faces of distinct people have large distances). Once this embedding is established the task of face verification becomes thresholding the distance between two embeddings – recognition becomes a k-NN classification problem. They train with triplet images with 2 positives and 1 negative and they train the loss to separate the positive pair from the negative by a distance margin. The network consists of batches as input a DCNN then L2 normalization 

RetinaFace: Single-stage Dense Face Localisation in the Wild 
https://arxiv.org/pdf/1905.00641v2.pdf

(1) We manually annotate five facial landmarks on the WIDER FACE dataset and observe significant improvement in hard face detection with the assistance of this extra supervision signal.
(2) We further add a self-supervised mesh decoder branch for predicting a pixel-wise 3D shape face information in parallel with the existing supervised branches.
(3) On the WIDER FACE hard test set, RetinaFace outperforms the state of the art average precision (AP) by 1.1% (achieving AP equal to 91.4%).
(4) On the IJB-C test set, RetinaFace enables state of the art methods (ArcFace) to improve their results in face verification (TAR=89.59% for FAR=1e-6). 
(5) By employing light-weight backbone networks, RetinaFace can run real-time on a single CPU core for a VGA-resolution image
5 facial landmarks 

***
# Extra

Improving face verification and person re-identification accuracy using hyperplane similarity
http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w23/Jones_Improving_Face_Verification_ICCV_2017_paper.pdf

This approach seeks to replace L2 distance comparisons with hyperplane similarity. 
Results show this improves verification results for low false acceptance rates 
This is based on the property that feature vectors for a person are linearly separable from feature vectors for all other people. 
To address this, they propose to compare feature vectors according to their distance to hyperplanes that separate one person’s feature vectors from all other person’s feature vectors (like we try to optimize in the training loss).
The idea is to use FV’s for a set of negative faces, which are faces from a variety of different people and compute hyperplanes between these faces and each of the two faces being compared. 
The sum of the margins to these hyperplanes from the feature vectors of the two test faces can be used in place of L2 distance. 
This similarity function is called hyperplane similarity. 
Negative faces can be from the training set or web. 
Hyperplane similarity allows us to achieve improvement in verification accuracy.
The drawback is speed, the straightforward way to is Linear SVM, they use a simpler algorithm.
A template is a set of one or more images of a particular person
Given a pair of templates P and Q, the goal of verification is to correctly classify whether they belong to the same identity or not. 
A template P can be represented by feature vectors Fp, and a hyperplane is calculated to discriminate Fp from Fn negative feature vectors







