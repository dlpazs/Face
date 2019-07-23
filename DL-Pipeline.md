# Face Recognition Pipeline
## Inspired by Full stack deep learning march 2019 [link](https://fullstackdeeplearning.com/march2019)

=> Planning & project setup : goals & requirements, resource allocation <=>
Data Collection & labelling : sensors, capture images, annotate with ground truth(how?) <=>
Training & debugging : implement a baseline in Opencv, Find SoTA model & reproduce, debug your implementation, improve model for task <=>
Deploying and testing : pilot, write tests to prevent regressions, roll out in production

NB: revisit metric, Performance in real world isn't great - revisit requirements (faster or more accurate)?

- Need to know SOTA in your domain? what are the hard problems, what are easy? What to try next? 

### Planning & project setup

- define project goals
- choose metric
- evaluate baselines
- setup codebase

forms of testing : preventing regressions in codebase (train model for single step and it still works), making sure if you train your entire model on smaller version of dataset you can achieve a certain loss, validation set has imbalance of what we care about. Does the validation really test what we care about, so a validation benchmark needs to reflect what we care about (maybe we collect more data of that, adding more data to model where the model doesn't do well so go back and collect more data for that manually).  

How often does it need to work to be useful?

What are we minimizing? 

Confusion Matrix
            pred
actual        +   -
          +   TP  FP
          -   FN  TN
          
FPR = FP / (FP + TN). TNR = 1 - FPR. FPR ~ 0 the better, FPR of 0 when FP = 0. Low FPR means high TNR meaning TNR ~1 and TNR=1 means FN = 0. If you optimize for Low FPR and Low TNR then you will have less FN and FP respectively. 
TPR = TP/(TP + FN). FNR = 1 - TPR. High TPR closer to 1 means FP = 0 and that means the FNR is closer to 0 and when FNR closer to 0 this makes FN = 0. 
TPR = Precision, FPR = Recall. 
Combining precision and recall of models (p + r) / 2. 

- Thresholding metric
choosing which metrics to threshold : Domain judgement (which metrics can you engineer around?), Which metrics are least sensitive to model choice?, Which metrics are closest to desired values? 
Choose threshold values : domain judgement (what is an acceptable tolerance? what performance is achieveable?), how well does the baseline do? How important is this metric right now? 

mean Average precision for precision and recall:
Average Precision (AP) = area under the curve. Mean over the classes to create mean AP. So we might want to use mAP for each classes existing in our DB of identities. 

A. Baselines give you lower bound on expected model performance:

B. tighter the lower bound, the more useful the baseline 

> What is human level performance? : Random people (Amazon Turk), etc.
> simple ml baselines : linear model with hand-crafted, basic NN models. 
> Is the compute available? 
> The goal isn't to get **SOTA** to ship a robust product
> 


## Tooling

#### Data 

- versioning dataset quilt, DVC, pachyderm. (L1 data lives of filesystem and db) (L2 data is versioned as assets nd code,  data stored as id with associated metadata) (L3 DVC creates S3 for data and link to S3)
- labelling [prodigy](https://prodi.gy/) [scale](https://scale.ai/) [hive](https://thehive.ai/) [supervisly](https://supervise.ly/)
- workflow trifacta, luigi
- database postgres, hadoop, mongo
- data lake (aggregation of data from multiple sources like db, logs, data transforms)
- binary data like images, sound, compressed texts should be stored as objects (S3)
- metadata (labels, user activity) stored in database
- [book](http://dataintensive.net/) [free](http://xfido.com/pdf/designing-data-intensive-applications.pdf)
- limitations of makefiles -> workflows -> airflow & Luigi (python) -> Airflow lets you define DAG with python -> TFX does this aswell
DAG graph spec that kicks of code, once that DAG is kicked off how do we distribute the resources : Airflow leaves it to rabbitMQ, so the workflow manager knows what to do and goes through the graph. RabbitMQ is a queing system, the workflow manager starts a queue for all the tasks and puts stuff on the queue and has workers that take stuff off the queue and put it back on or not if it fails. 
- S3 simple storage service : object storage (abstraction of filesystem, you put data in this id path and get data from id path. And S3 stores it in many places, versions it.) 

#### Development/Training/Eval

- SE : git, python, jupyter, frameworks, resource management docker k8's, experiment management TensorBoard, weights and biases, distributed training, hyperparam optimization SIGOPT). 

#### Deployment

- Cloud / embed/ mobile
- Ci/Testing, jenkins, buildkite, 
- web : k8's algorithmia, 
- monitoring : ? nothing to monitor model predictions. 
- interchange : ONNX
- hardware/mobile
- [domino](https://www.dominodatalab.com/) 
- convert pytorch to onnx to caffe2 for super optimisation [here](https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html)



## Debugging

- Incorrect shapes
- pre-process input incorrectly
- wrong input to your loss e.g. softmaxed ouputs to a loss that expects logits  
- setting training mode for batch norm
- numerical instability i.e. nan
- Start with lightweight implementation < 200 lines of code in single file
- use off-shelf components e.g. keras
- start with dataset you can load into memory, before complex data pipeline
- get your model to run : shape mismatch casting issue
- stepping in debugger use ipdb in pytorch i.e. `import ipdb; ipdb.set_trace()`
- tensorflow use `tfdb`
- Out-of-memory issues : smaller batch size, too large fc layers, too much data, duplicating ops, tf.data api is good at splitting large data into chunks and feeding it as tensors. 
- Try overfit a single batch of data as close to 0 : error goes up (flip sign of loss/grad, high lr, softmax taken over wrong dim), error explodes (numerical issue, check all exp/log/div) error oscilliates (lr), error plateaus (lr too low, grad not flowing through whole model, too much reg, incorrect input to loss, data / labels corrupted). 
- Compare to known result : More useful (official model implementation evaluated on similar dataset to yours - walk code line-by-line and ensure you have same output) (unofficial model implementation). Compare with results of paper. 
- iteration speed 

- Error analysis: test-val set errors, train-val set errors, 

### [Weights & Biases](https://docs.wandb.com/docs/started.html)

- Track and Visualize ML experiments

### Testing and deployment

- Training system (process raw data, run experiments, manage results) -> Prediction system (process input, construct network, make preds) -> serving system (serve predictions, scale to demand) -> validation tests (test pred system on validation set, start from processed data, runs every hour /code push, cataches model regressions) -> functionality tests (test pred system on few important examples, runs quick, catches code regressions) -> monitoring (alert to downtime, errors, distribution shifts

#### ML Test Score: (rubric for ml production readiness)

Data test as well as model testing
- feature expectations captured in schema
- all features beneficial
- no features cost too much
- data pipeline has appropriate privacy controls
- new features added quickly
- all input feature code is tested

Model tests
- model specs reveiwed
- offline and online metrics correlate
- hyperparams tuned
- simpler model is not better
- model quality

infrarstructure tests
- training reproducible
- model specs unit tested
- ml pipeline is integration tested
- model quality validated before serving
- model is debuggable
- serving models can be rolled back

monitoring tests
- dependency changes result in notification
- data invariants hold for inputs
- training/serving not skewed
- numerical stability
- computing performance not regressed
- pred qualtiy not regressed

Unit /integration tests
- tests individual module and whole system
CI 
- tets run every time new code pushed to repo before updated model deployed
SaaS for CI
- jenkins
Containerization (Docker)
- self-enclosed environment for running tests

#### Deploying to cloud instances

- lots of instances because we want to scale to many requests and each can only do so many requests. We put all of it behind a load-balancer (http interface that user hits that IP and load-balancer says I have these 5 instances and randomly send a request to an instance, it pings instances to check they're healthy, if they're all super busy it can add more through auto-scaling). 

#### Deploy docker to cloud

- app code/dependencies packages to orchestrator like K8's 
- cons: still pay for servers/ pay for uptime few requests

#### Deploy serverless functions

- code weights and we can package it into .zip files , with a single entry point function
- then AWS Lambda/ GCP manage everything else: instant scaling to 10,000+ requests per second, load balancing. 
- only pay for compute town and lowers dev ops load
- cons: entire deployment package has to fit within 500MB <5 min execution, <3gb memory on lambda
- Easy to have two versions of Lambda functions in production and start sending low volume traffic to one

#### Model Serving

- tf serving : TF preds with high throughput using GCPML. Overkill unless why you need, unless CPU is too slow
- model server mxnet : overkill with load balancers etc
- Clipper : open source using rest using docker with load balancing/ versioning / state of the are bandit and ensemble methods to select and combine predictions (multiple models in prod and every input goes through all of them and something ontop of them ensembling prediction out and doing majority vote).
- algorithmia : train model, git push into ai layer, makes model to scale/manage hardware and make it an API (overkill)
- CPU inference maybe go serverless or use load balancer, can be deploy docker as easily as lambda

- pipenv lock --requirements --keep-outdates which creates a requirements.txt file

### Considerations

- Recognition Rate - relies on list of gallery images (one per identity) and a list of probe images of the same identities. For each
probe image the similarity to all gallery images is computed and it's determined if the gallery image with the highest similarity
i.e. lowest distance value is from the same identity as the probe image. The Recognition Rate is the total number of correctly
identified probe images, divided by the total number of probe images. 

- Verification Rate - list of image pairs, where pair with the same and pairs of different identities are compared. Given the lists of similarities
of both, the Receiver Operating Characteristics can be computed and the verification rate. (So triplet loss -> ROC : plots TPR against FPR at different
threshold's. TPR = sensitivity, recall. FPR = fallout, false-alarm. Sensitivity measures the proportion of actual positives that are correctly
classified. Specificity measures the proportion of actual negatives that are correctly identified)

