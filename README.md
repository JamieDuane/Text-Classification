## Task Description:
This task is to apply TensorFlow and Keras packages to conduct text classification for sentiment analysis. It includes 4 steps:

Step1: import and preprocess dataset
Step2: embedding with GloVe
Step3: define and train model
Step4: test and evaluate model

There are three models. By default, CNN is applied here since it is proven to perform well for document classification. These models can be selected in 'Run.py' at:
line 104 (Simple Binary Classification), 
line 107 (Convolutional Neural Network), 
line 110 (Recurrent Neural Network).


## Step1:
a. Import 'train.json' as training data, 'test.json' as testing data, 'dev.json' as validation data;

b. Tokenize 3 datasets by package 'Tokenizer' from 'tensorflow.keras.preprocessing.text';

c. Convert them to be sequences and 'padded' them with pad_sequences from 'tensorflow.keras.preprocessing.sequence';


## Step2:
a. Load pre-trained embeddings from 'glove.840B.300d.txt';

b. Vectorize all the tokens in training data and create embedding layer for model definition;


## Step3:
a. Define model with 'Sequential' from 'tensorflow.keras.models':
      
      i. Model 1: simple binary classification model;
	 
     ii. Model 2: Convolutional Neural Network model;
	
    iii. Model 3: Recurrent Neural Network model;

b. Compile model and record checkpoint;
c. Define 2 checkpoints for outcome needed:

      i. Checkpoint 1: save weights of model according to max training accuracy if it is less than 90%;

     ii. Checkpoint 2: save weights of model according to max validation accuracy if training accuracy is larger than 90%

d. Fit model with validation_dataset='dev.json', epochs=10;


## Step4:
a. Load weight of the best training accuracy;
b. Predict probabilities and classes on testing data;
c. Applied predicted value to come with precision, recall, arc, loss and accuracy;


## Output:
1. For each epoch, the output is like:
   
       Epoch 3/10
       270/270 [==============================] - 4s 14ms/step - loss: 0.3753 - accuracy: 0.8366 - val_loss: 0.5069 - val_accuracy: 0.7510
       Epoch 00001: accuracy improved from 0.78879 to 0.83661, saving model to tmp
       .
	    .
	    .
       Epoch 7/10 
       Training accuracy is larger than 0.9
       Now it is going to pick out the best validation accuracy
       270/270 [==============================] - 4s 13ms/step - loss: 0.0686 - accuracy: 0.9771 - val_loss: 0.9631 - val_accuracy: 0.7260
       Epoch 00001: val_accuracy did not improve from 0.74375
	
2. For evaluation of this model, the output is like:
   
       Test Loss      :      0.988353
       Test Accuracy  :      72.983116%
       Test Precision :      77.777778%
       Test Recall    :      64.352720%
       Test AUC       :      0.810811
       Test Confusion Matrix:[[435  98]
                              [190 343]]


## Brief analysis of result:
Training accuracy can be up to 98% with epochs equal to 10, while the validation accuracy floats up and down. Somehow the model is overfit, so Dropout layer is applied to reduce overfitting.

Test accuracy can be up to 70% and sometimes can be up to 75%. More precise test accuracy can be obtained by running program by more times to find average test accuracy. The model is not reliable enough to predict real world situation.
It may be because of small size of training data or preprocess session have not been conducted properly and thoroughly.

