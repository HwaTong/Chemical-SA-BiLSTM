def train_model(model_path, nb_classes, gram_len, vocab_len, X_train, Y_train, X_valid, Y_valid, nb_epoch):

    callbacks_list = [ModelCheckpoint(model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True),
                      CSVLogger(filename=model_path + '_train.log', append=True)
                      ]

    model=Sequential()
    model.add(Masking(mask_value=0., input_shape=(MAXLEN - gram_len + 1, vocab_len + 7)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))

    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(attention_flatten(32))
    model.add(Dropout(0.35))
    model.add(Dense(units=256,activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(units=nb_classes, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    logging.info('Training the model')

    history = model.fit(X_train, Y_train, epochs=nb_epoch, validation_data=(X_valid, Y_valid), verbose=2, callbacks=callbacks_list)

    np.savez_compressed(model_path + '_history.npz', tr_acc=history.history['accuracy'], tr_loss=history.history['loss'],
                        val_acc=history.history['val_accuracy'], val_loss=history.history['val_loss'])

    loss_train = min(history.history['loss'])
    accuracy_train = max(history.history['accuracy'])

    logging.info("Training Loss: %f" % loss_train)
    logging.info("Training Accuracy: %f" % accuracy_train)
    print('\nLog Loss and Accuracy on Train Dataset:')
    print("Loss: {}".format(loss_train))
    print("Accuracy: {}".format(accuracy_train))
    print()

    loss_val = min(history.history['val_loss'])
    accuracy_val = max(history.history['val_accuracy'])

    logging.info("Validation Loss: %f" % loss_val)
    logging.info("Validation Accuracy: %f" % accuracy_val)
    print('\nLog Loss and Accuracy on Val Dataset:')
    print("Loss: {}".format(loss_val))
    print("Accuracy: {}".format(accuracy_val))
    print()

    plt.clf()
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.ylim(ymin=0.6, ymax=1.1)
    plt.savefig(model_path + "_accuracy.png", type="png", dpi=300)

    plt.clf()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(model_path + "_loss.png", type="png", dpi=300)

    plt.clf()