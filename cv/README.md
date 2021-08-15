# Computer Vision 


## Supervised Learning

### Training Strategies : 
* Used a small CNN based network shown below.
![image](https://user-images.githubusercontent.com/57453637/129484770-c4ed52b0-ab18-4987-85a8-9045a7cb6a53.png)
* The network was very sensitive to small changes in dropout probabilities. 
* Since the dataset was small, I didn't divide into train/dev set. I used the test set to tune the hyperparameters (this is a really bad practice). 
* Hyperparams used :  BATCH_SIZE = 64  LR = 0.001 EPOCHS = 100



### Results : 


Train Accuracy ~ 84%


Test Accuracy ~ 64%
            
Conclusions : Since the dataset as a whole had only 5000 examples, the model faces overfitting. One possible way to increase the test accuracy is by having more training examples



## Pseudo Labelling : 
- Since there are memory limits in Google Colab and I don't have a CUDA compatible GPU, I havent been able to complete this part. The code has been written in reference to the blog post given.
### Training Strategies : 
* Used the same network as above.
* I followed everything given in the reference blogpost.


## SimCLR :
- I have refered this [blog](https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/) and was facing an error in the end. Due to time constraints, I was not able to complete. 
### Training Strategies : 
* Implementation using Pytorch-Lightning as given in the above blog.
* The similarity function used is cosine similarity
* Augmentations used are Random Resized Crop, Random Resized Rotation, Color Jitter, and Random Grayscale. These were some of the recommended transformations from the paper. 
* Since the blog had good results using Efficient net, this is the architecture which  I have also used.
* The temperature parameter has been kept to 0.5 as given in the paper.



### When can we not use semi supervised learning ? 
Unless the learner is absolutely certain there is some non-trivial relationship between labels and the unlabeled distribution (“SSL type assumption”), semi-supervised learning cannot provide significant advantages over supervised learning. The sample complexity of SSL is no more than a constant factor better than SL for any unlabeled distribution, under a no-prior-knowledge setting (i.e. without SSL type assumptions).


References : 
- http://www.cs.toronto.edu/~tl/papers/lumastersthesis.pdf

