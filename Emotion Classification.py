#!/usr/bin/env python


import sys, os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/source/')


import source.classification  as cls




'''
The following function will be called to train and test your model.
The function name, signature and output type is fixed.
The first argument is file name that contain data for training.
The second argument is file name that contain data for test.
The function must return predicted values or emotion for each data in test dataset
sequentially in a list.
['sad', 'happy', 'fear', 'fear', ... , 'happy']
'''

def  aithon_level2_api(traingcsv, testcsv):
    import keras
    from keras.utils import np_utils
    import sklearn
    from keras.preprocessing.image import ImageDataGenerator
    y_labels = traingcsv['emotion'].replace(['Fear','Sad','Happy'],[0,1,2]).values.astype('float32')
    y_train = keras.utils.np_utils.to_categorical(y_labels,3)
    X_train = traingcsv.iloc[:,1:].values.astype('float32')
    X_test = testcsv.values.astype('float32')
    X_train = X_train.reshape(X_train.shape[0],48,48,1)
    X_test = X_test.reshape(X_test.shape[0],48,48,1)



    n_train = len(y_train)
    n_test = len(X_test)
    sample_weights = sklearn.utils.class_weight.compute_sample_weight('balanced', y_labels[:n_train])

    # 


    
    def myfunc2(image):
        image = skimage.exposure.equalize_hist(image, nbins = 256)
        return image

    # 
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,fill_mode='nearest')

    # 


    # 
    test_datagen = ImageDataGenerator(horizontal_flip = True)

    # 
    

    # 
    train_gen = train_datagen.flow(x = X_train,y = y_train,batch_size = 64, shuffle = True, sample_weight = sample_weights)
    test_gen = test_datagen.flow(x = X_test,batch_size = 1, shuffle = False)
 
    # 
    import keras
    from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Dropout
    from keras.losses import categorical_crossentropy
    from keras.optimizers import Adam, RMSprop
    from keras.regularizers import l2
    from keras import Sequential
    from keras.layers import Conv2D, Flatten, Dense, Activation


    # 
    from keras import regularizers
    from keras import optimizers
    

    # 
    model = Sequential()    
    model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same',input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.2))


    model.add(Conv2D(128, (5, 5), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.2))
          
    model.add(Conv2D(512, (3, 3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.2))          
  
    model.add(Conv2D(512, (3, 3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.2))          

    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.2))    

    #model.add(Dense(512))
    #   model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))      

    model.add(Dense(3, activation='softmax'))
    model.compile(loss= 'categorical_crossentropy',
              optimizer=Adam(lr = 0.0005),
              metrics=['accuracy'])
    checkpoint = keras.callbacks.ModelCheckpoint(filepath='weights.h5', save_best_only=True, monitor = 'val_accuracy', mode = 'max')
    from keras.callbacks import ReduceLROnPlateau
    from keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 2, min_lr = 0.00001, mode = 'auto')
    callbacks = [checkpoint, reduce_lr, es]
    epochs = 1000
    steps_per_epoch = n_train//64

    model.fit_generator(steps_per_epoch = steps_per_epoch,epochs = epochs,generator = train_gen, callbacks = callbacks)
     
    arr = model.predict_classes(x = test_gen)


    # 
    arr = pd.Series(arr).replace([0,1,2],['fear','sad','happy']).values

    # 
    List = []
    for i in arr:
        List.append(i)

    # 
    return List
    
    
    
    
