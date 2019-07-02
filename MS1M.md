# MS1M 

[Link](https://arxiv.org/pdf/1607.08221.pdf)

- A new benchmark task to link recognizing face images to their corresponding entity keys in a knowledge base
- 1M celebrity images by using all collected photos of their face
- 10M images
- aims to overcome face recognition with disambiguation
- it answers the questions if it is "X" in an image which "X" not just who is in the image
- First, they define face recognition as to determine the identity of a person from their images
- Also introduce a knowledge base into face recognition, since knowledge bases have provided remarkable accuracy as identifiers
- Their task is to recognize a face image then link it with the corresponding entity key in the knowledge base
- **linking the image with an entity key in the knowledge base, rather than an isolated string for a person's name naturally
solves the disambiguation issue** in traditional face recognition
- The linked entity key is associated with rich and comprehensive property information in the knowledge base
- The scale of the problem introduces attractive challenges. With the increased number of classes, the inter-class variance decreases
- There are some celebs that look similar/twins
- large intra-class variance is introduced by popular celebs with millions of images available, as well as celebs with large appearance
variation
- **Face verification** is determining whether two given face images belong to the same person
- **LFW** is the most widely used measurement for verification with 3000 matched face image pairs and 3000 mismatched face image pairs
- More Recently, another task called **Face Identification** involves, two sets of face images (a gallery set and query set). The task
is for a given gave image in the query set, to find the most similar faces in the gallery image set
- When the gallery set has only a limited number (<5) of face images for each individual, the most effective solution is to learn a
generic feature which can tell whether or not two face images are the same person. Which is still essentially the same problem as 
face verification
- MegaFace is the most difficult face identification benchmark
- The problem comes with 1 million distractors blended in the gallery image set
