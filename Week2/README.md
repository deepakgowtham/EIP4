Epochs:

Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 37s 618us/step - loss: 0.2348 - acc: 0.9261 - val_loss: 0.0532 - val_acc: 0.9819
Epoch 2/20
60000/60000 [==============================] - 35s 577us/step - loss: 0.0753 - acc: 0.9767 - val_loss: 0.0344 - val_acc: 0.9891
Epoch 3/20
60000/60000 [==============================] - 35s 578us/step - loss: 0.0585 - acc: 0.9814 - val_loss: 0.0333 - val_acc: 0.9893
Epoch 4/20
60000/60000 [==============================] - 35s 578us/step - loss: 0.0500 - acc: 0.9843 - val_loss: 0.0322 - val_acc: 0.9899
Epoch 5/20
60000/60000 [==============================] - 35s 577us/step - loss: 0.0442 - acc: 0.9863 - val_loss: 0.0293 - val_acc: 0.9906
Epoch 6/20
60000/60000 [==============================] - 34s 574us/step - loss: 0.0409 - acc: 0.9874 - val_loss: 0.0332 - val_acc: 0.9896
Epoch 7/20
60000/60000 [==============================] - 35s 578us/step - loss: 0.0388 - acc: 0.9880 - val_loss: 0.0284 - val_acc: 0.9912
Epoch 8/20
60000/60000 [==============================] - 35s 576us/step - loss: 0.0354 - acc: 0.9887 - val_loss: 0.0252 - val_acc: 0.9914
Epoch 9/20
60000/60000 [==============================] - 35s 577us/step - loss: 0.0336 - acc: 0.9895 - val_loss: 0.0284 - val_acc: 0.9909
Epoch 10/20
60000/60000 [==============================] - 35s 584us/step - loss: 0.0318 - acc: 0.9899 - val_loss: 0.0240 - val_acc: 0.9922
Epoch 11/20
60000/60000 [==============================] - 35s 576us/step - loss: 0.0289 - acc: 0.9906 - val_loss: 0.0284 - val_acc: 0.9912
Epoch 12/20
60000/60000 [==============================] - 35s 578us/step - loss: 0.0298 - acc: 0.9906 - val_loss: 0.0249 - val_acc: 0.9923
Epoch 13/20
60000/60000 [==============================] - 35s 580us/step - loss: 0.0278 - acc: 0.9912 - val_loss: 0.0254 - val_acc: 0.9919
Epoch 14/20
60000/60000 [==============================] - 35s 576us/step - loss: 0.0263 - acc: 0.9916 - val_loss: 0.0252 - val_acc: 0.9927
Epoch 15/20
60000/60000 [==============================] - 34s 570us/step - loss: 0.0277 - acc: 0.9910 - val_loss: 0.0215 - val_acc: 0.9934
Epoch 16/20
60000/60000 [==============================] - 34s 573us/step - loss: 0.0242 - acc: 0.9920 - val_loss: 0.0233 - val_acc: 0.9932
Epoch 17/20
60000/60000 [==============================] - 35s 579us/step - loss: 0.0237 - acc: 0.9919 - val_loss: 0.0225 - val_acc: 0.9930
Epoch 18/20
60000/60000 [==============================] - 35s 577us/step - loss: 0.0238 - acc: 0.9926 - val_loss: 0.0197 - val_acc: 0.9942
Epoch 19/20
60000/60000 [==============================] - 35s 582us/step - loss: 0.0230 - acc: 0.9925 - val_loss: 0.0216 - val_acc: 0.9927
Epoch 20/20
60000/60000 [==============================] - 34s 572us/step - loss: 0.0215 - acc: 0.9928 - val_loss: 0.0220 - val_acc: 0.9925

<keras.callbacks.History at 0x7f9fd793a2b0>

Model.Evaluvate:

[0.022001611612652777, 0.9925]

Strategy:

On observing the eight notebooks, I made an assumption that the network with global receptive field of more than 18 was achieving good results, so I designed my network with that assumption.
In order to reduce the parameters to less than 15k as discussed, I reduced the number of kernels used at each layer. 
I also noticed that the last layer where we use 7x7 convolution was consuming more parameters so added an extra layer of 3x3 convolution and used a 5x5 convolution to reduce the total parameters used.
Increasing the number of kernels at 1x1 convolution layer helped to achieve better performance.
When I finalized the network parameters, the max train accuracy was around 99.9% and max validation accuracy was around 99.2% so assuming overfitting added normalization and dropout to improve the validation accuracy. 
