#Final Accuracy on Base Network
Accuracy on test data is: 82.29


#Model Definition
model_2= Sequential()
model_2.add(SeparableConv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3) )) # 30, 3
model_2.add(BatchNormalization())
model_2.add(Dropout(rate=0.1))

model_2.add(SeparableConv2D(64, (3, 3), activation='relu' ))  #28, 5
model_2.add(BatchNormalization())
model_2.add(Dropout(rate=0.1))
model_2.add(SeparableConv2D(64, (3, 3), activation='relu' ))  #26, 7
model_2.add(BatchNormalization())
model_2.add(Dropout(rate=0.1))
model_2.add(SeparableConv2D(128, (3, 3), activation='relu' )) # 24, 9
model_2.add(BatchNormalization())
model_2.add(Dropout(rate=0.1))
model_2.add(SeparableConv2D(128, (3, 3), activation='relu' )) # 22, 11
model_2.add(BatchNormalization())
model_2.add(Dropout(rate=0.1))


model_2.add(MaxPooling2D(2))                                    #11, 12
model_2.add(Convolution2D(80, (1, 1), activation='relu' ))       #11, 12
model_2.add(BatchNormalization())
model_2.add(Dropout(rate=0.1))

model_2.add(SeparableConv2D(128, (3, 3), activation='relu' ))  #9, 16
model_2.add(BatchNormalization())
model_2.add(Dropout(rate=0.1))
model_2.add(SeparableConv2D(128, (3, 3), activation='relu' )) # 7, 20
model_2.add(BatchNormalization())
model_2.add(Dropout(rate=0.1))
model_2.add(SeparableConv2D(128, (3, 3), activation='relu' )) # 5, 24
model_2.add(BatchNormalization())
model_2.add(Dropout(rate=0.1))

model_2.add(SeparableConv2D(10, (5,5), activation='relu'))    #1, 32
model_2.add(Flatten())
model_2.add(Activation('softmax'))


#Epoch Logs

Epoch 1/50
  1/781 [..............................] - ETA: 2:15 - loss: 1.4406 - acc: 0.5000

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:11: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  # This is added back by InteractiveShellApp.init_path()
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:11: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, workers=-1, steps_per_epoch=781, epochs=50)`
  # This is added back by InteractiveShellApp.init_path()

781/781 [==============================] - 65s 83ms/step - loss: 1.3250 - acc: 0.5330 - val_loss: 1.2076 - val_acc: 0.5808
Epoch 2/50
781/781 [==============================] - 64s 82ms/step - loss: 1.1493 - acc: 0.5943 - val_loss: 1.0227 - val_acc: 0.6447
Epoch 3/50
781/781 [==============================] - 64s 82ms/step - loss: 1.0517 - acc: 0.6310 - val_loss: 0.9458 - val_acc: 0.6725
Epoch 4/50
781/781 [==============================] - 64s 82ms/step - loss: 0.9861 - acc: 0.6520 - val_loss: 0.8418 - val_acc: 0.7044
Epoch 5/50
781/781 [==============================] - 64s 82ms/step - loss: 0.9392 - acc: 0.6711 - val_loss: 0.8383 - val_acc: 0.7103
Epoch 6/50
781/781 [==============================] - 64s 82ms/step - loss: 0.9085 - acc: 0.6822 - val_loss: 0.8627 - val_acc: 0.7055
Epoch 7/50
781/781 [==============================] - 64s 82ms/step - loss: 0.8771 - acc: 0.6943 - val_loss: 0.7843 - val_acc: 0.7248
Epoch 8/50
781/781 [==============================] - 64s 82ms/step - loss: 0.8549 - acc: 0.7011 - val_loss: 0.7776 - val_acc: 0.7310
Epoch 9/50
781/781 [==============================] - 64s 82ms/step - loss: 0.8270 - acc: 0.7098 - val_loss: 0.7243 - val_acc: 0.7495
Epoch 10/50
781/781 [==============================] - 64s 82ms/step - loss: 0.8024 - acc: 0.7185 - val_loss: 0.7563 - val_acc: 0.7423
Epoch 11/50
781/781 [==============================] - 64s 82ms/step - loss: 0.7868 - acc: 0.7243 - val_loss: 0.6916 - val_acc: 0.7627
Epoch 12/50
781/781 [==============================] - 64s 82ms/step - loss: 0.7736 - acc: 0.7300 - val_loss: 0.7527 - val_acc: 0.7490
Epoch 13/50
781/781 [==============================] - 64s 82ms/step - loss: 0.7658 - acc: 0.7333 - val_loss: 0.7412 - val_acc: 0.7457
Epoch 14/50
781/781 [==============================] - 64s 82ms/step - loss: 0.7472 - acc: 0.7374 - val_loss: 0.6752 - val_acc: 0.7664
Epoch 15/50
781/781 [==============================] - 64s 82ms/step - loss: 0.7381 - acc: 0.7432 - val_loss: 0.6820 - val_acc: 0.7649
Epoch 16/50
781/781 [==============================] - 64s 82ms/step - loss: 0.7201 - acc: 0.7500 - val_loss: 0.6699 - val_acc: 0.7685
Epoch 17/50
781/781 [==============================] - 64s 82ms/step - loss: 0.7157 - acc: 0.7488 - val_loss: 0.6362 - val_acc: 0.7805
Epoch 18/50
781/781 [==============================] - 64s 81ms/step - loss: 0.7036 - acc: 0.7538 - val_loss: 0.6442 - val_acc: 0.7787
Epoch 19/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6942 - acc: 0.7572 - val_loss: 0.6115 - val_acc: 0.7898
Epoch 20/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6838 - acc: 0.7594 - val_loss: 0.6482 - val_acc: 0.7758
Epoch 21/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6739 - acc: 0.7653 - val_loss: 0.6223 - val_acc: 0.7899
Epoch 22/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6644 - acc: 0.7706 - val_loss: 0.6586 - val_acc: 0.7775
Epoch 23/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6600 - acc: 0.7688 - val_loss: 0.6246 - val_acc: 0.7831
Epoch 24/50
781/781 [==============================] - 64s 81ms/step - loss: 0.6543 - acc: 0.7706 - val_loss: 0.6305 - val_acc: 0.7861
Epoch 25/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6501 - acc: 0.7732 - val_loss: 0.6254 - val_acc: 0.7874
Epoch 26/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6439 - acc: 0.7759 - val_loss: 0.5611 - val_acc: 0.8071
Epoch 27/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6296 - acc: 0.7799 - val_loss: 0.6160 - val_acc: 0.7886
Epoch 28/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6307 - acc: 0.7778 - val_loss: 0.5830 - val_acc: 0.8011
Epoch 29/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6208 - acc: 0.7849 - val_loss: 0.5774 - val_acc: 0.7986
Epoch 30/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6173 - acc: 0.7846 - val_loss: 0.5649 - val_acc: 0.8050
Epoch 31/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6115 - acc: 0.7867 - val_loss: 0.5819 - val_acc: 0.8058
Epoch 32/50
781/781 [==============================] - 64s 82ms/step - loss: 0.6097 - acc: 0.7870 - val_loss: 0.5783 - val_acc: 0.8041
Epoch 33/50
781/781 [==============================] - 64s 82ms/step - loss: 0.5993 - acc: 0.7908 - val_loss: 0.5526 - val_acc: 0.8133
Epoch 34/50
781/781 [==============================] - 63s 81ms/step - loss: 0.5959 - acc: 0.7926 - val_loss: 0.5606 - val_acc: 0.8066
Epoch 35/50
781/781 [==============================] - 64s 81ms/step - loss: 0.5868 - acc: 0.7957 - val_loss: 0.5744 - val_acc: 0.8011
Epoch 36/50
781/781 [==============================] - 63s 81ms/step - loss: 0.5835 - acc: 0.7959 - val_loss: 0.5706 - val_acc: 0.8060
Epoch 37/50
781/781 [==============================] - 64s 82ms/step - loss: 0.5835 - acc: 0.7972 - val_loss: 0.5457 - val_acc: 0.8136
Epoch 38/50
781/781 [==============================] - 66s 85ms/step - loss: 0.5803 - acc: 0.7984 - val_loss: 0.6529 - val_acc: 0.7788
Epoch 39/50
781/781 [==============================] - 64s 82ms/step - loss: 0.5750 - acc: 0.7994 - val_loss: 0.5543 - val_acc: 0.8155
Epoch 40/50
781/781 [==============================] - 64s 82ms/step - loss: 0.5695 - acc: 0.8015 - val_loss: 0.5666 - val_acc: 0.8077
Epoch 41/50
781/781 [==============================] - 63s 81ms/step - loss: 0.5633 - acc: 0.8017 - val_loss: 0.5352 - val_acc: 0.8203
Epoch 42/50
781/781 [==============================] - 63s 81ms/step - loss: 0.5666 - acc: 0.8034 - val_loss: 0.5383 - val_acc: 0.8158
Epoch 43/50
781/781 [==============================] - 64s 81ms/step - loss: 0.5582 - acc: 0.8034 - val_loss: 0.5470 - val_acc: 0.8140
Epoch 44/50
781/781 [==============================] - 63s 81ms/step - loss: 0.5542 - acc: 0.8057 - val_loss: 0.5442 - val_acc: 0.8160
Epoch 45/50
781/781 [==============================] - 63s 81ms/step - loss: 0.5515 - acc: 0.8073 - val_loss: 0.5652 - val_acc: 0.8083
Epoch 46/50
781/781 [==============================] - 63s 81ms/step - loss: 0.5526 - acc: 0.8061 - val_loss: 0.5519 - val_acc: 0.8120
Epoch 47/50
781/781 [==============================] - 64s 81ms/step - loss: 0.5421 - acc: 0.8096 - val_loss: 0.4976 - val_acc: 0.8307
Epoch 48/50
781/781 [==============================] - 64s 82ms/step - loss: 0.5426 - acc: 0.8105 - val_loss: 0.5238 - val_acc: 0.8259
Epoch 49/50
781/781 [==============================] - 64s 82ms/step - loss: 0.5409 - acc: 0.8088 - val_loss: 0.5439 - val_acc: 0.8152
Epoch 50/50
781/781 [==============================] - 64s 82ms/step - loss: 0.5358 - acc: 0.8121 - val_loss: 0.5049 - val_acc: 0.8259
Model took 3194.11 seconds to train
