import csv
import itertools
import time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
from sklearn.utils import shuffle
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.layers import Masking, Layer, InputSpec, Dropout
from keras.layers import LSTM, Bidirectional
from keras.utils import Sequence, to_categorical
from keras_self_attention import SeqSelfAttention

physical_devices = tf.config.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


DATA_PATH = 'data/'
MAXLEN = 1002
BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}
CODES = 'ACDEFGHIKLMNPQRSTVWY'

def main():
    nb_epoch = 100
    batch_size = 128
    encoding = 'oh'
    subontologies = ['bp', 'mf', 'cc']
    gram_len = 1
    runs = range(1, 11)


    with open('Chemical_SA_BiLSTM_' + str(nb_epoch) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["subontology", "run", "f", "p", "r", "a","train_time", "predict_time"])

        for subontology in subontologies:
            subontology_df = pd.read_pickle(DATA_PATH + subontology + '.pkl')
            subontologies = subontology_df['functions'].values

            nb_classes = len(subontologies)

            train_df, valid_df, test_df = load_data(subontology)
            train_data, train_labels = get_data(train_df)
            val_data, val_labels = get_data(valid_df)
            test_data, test_labels = get_data(test_df)

            vocab = {}
            for index, gram in enumerate(itertools.product(CODES, repeat=gram_len)):
                vocab[''.join(gram)] = index + 1

            for run in runs:
                tf.keras.backend.clear_session()

                model_path = 'models/tpe_exp_' + encoding + '_' + subontology + '_e' + str(nb_epoch) + '_b' + str(
                    batch_size) + '_n' + str(gram_len) + '_v' + '_r' + str(run)
                logging.basicConfig(filename=model_path + '.log', format='%(message)s', filemode='w',
                                    level=logging.INFO)
                logging.info('Model: %s' % model_path)
                logging.info('Subontologies: %s %d' % (subontology, len(subontologies)))

                train_data, train_labels = shuffle(train_data, train_labels)
                val_data, val_labels = shuffle(val_data, val_labels)
                test_data, test_labels = shuffle(test_data, test_labels)

                X_train, Y_train = createdata(train_data, train_labels, vocab, gram_len)
                X_valid, Y_valid = createdata(val_data, val_labels, vocab, gram_len)
                X_test, Y_test = createdata(test_data, test_labels, vocab, gram_len)

                start_time = time.time()

                train_model(model_path=model_path, nb_classes=nb_classes, gram_len=gram_len, vocab_len=len(vocab),
                            X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, nb_epoch=nb_epoch)

                train_time = time.time() - start_time
                print()
                print("--- train_time %s seconds ---" % train_time)
                print()

                start_time = time.time()

                predictions = predict(model_path, X_test)

                predict_time = time.time() - start_time
                print()
                print("--- predict_time %s seconds ---" % predict_time)
                print()

                logging.info('Evaluation on test data')
                print('Evaluation on test data')
                f, p, r, a = compute_performance(predictions, test_labels)
                logging.info('Fmax\t\tPrecision\tRecall\n%f\t%f\t%f' % (f, p, r))
                print('Fmax\t\tPrecision\tRecall\n%f\t%f\t%f' % (f, p, r))
                print('accuracy:', a)

                writer.writerow([subontology, run, f, p, r, a, train_time, predict_time])

def getAminoAcidCharge(x):
    if x in "KR":
        return 1.0
    if x == "H":
        return 0.1
    if x in "DE":
        return -1.0
    return 0.0

def getAminoAcidHydrophobicity(x):
    AminoAcids = "ACDEFGHIKLMNPQRSTVWY"
    _hydro = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2,-0.9, -1.3]
    return _hydro[AminoAcids.find(x)]

def isAminoAcidPolar(x):
    return x in "DEHKNQRSTY"

def isAminoAcidAromatic(x):
    return x in "FWY"

def hasAminoAcidHydroxyl(x):
    return x in "ST"

def hasAminoAcidSulfur(x):
    return x in "CM"

def createdata(inputs, targets,vocab, gram_len):
    index = np.arange(len(inputs))
    batch = [inputs[k] for k in index]
    labels = np.asarray([targets[k] for k in index])
    ngrams = list()
    np_prots = list()
    for seq in batch:
        grams = np.zeros((len(seq) - gram_len + 1,), dtype='int32')
        np_prot = list()
        for i in range(len(seq) - gram_len + 1):
            a = seq[i]
            descArray = [float(x) for x in [getAminoAcidCharge(a), getAminoAcidHydrophobicity(a), isAminoAcidPolar(a),isAminoAcidAromatic(a), hasAminoAcidHydroxyl(a), hasAminoAcidSulfur(a)]]
            np_prot.append(descArray)
            grams[i] = vocab[seq[i: (i + gram_len)]]
        np_prots.append(np_prot)
        ngrams.append(grams)
    np_prots = sequence.pad_sequences(np_prots, maxlen=MAXLEN)
    ngrams = sequence.pad_sequences(ngrams, maxlen=MAXLEN)
    res_inputs = to_categorical(ngrams, num_classes=len(vocab) + 1)
    res_inputs = np.concatenate((res_inputs, np_prots), 2)
    return res_inputs,labels

@tf.keras.utils.register_keras_serializable()
class KMaxPooling(Layer):
    def __init__(self, k=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[1] * self.k))

    def call(self, top_k):
        top_k = tf.transpose(top_k, [0, 2, 1])
        top_k = tf.nn.top_k(top_k, k=self.k, sorted=True, name=None)[0]
        top_k = tf.transpose(top_k, [0, 2, 1])
        return top_k

    def get_config(self):
        config = super(KMaxPooling, self).get_config()
        config.update({"k": self.k})
        return config

def load_data(subontology):
    df = pd.read_pickle(DATA_PATH + 'train' + '-' + subontology + '.pkl')
    n = len(df)
    index = df.index.values

    valid_n = int(n * 0.875)
    train_df = df.loc[index[:valid_n]]
    valid_df = df.loc[index[valid_n:]]

    test_df = pd.read_pickle(DATA_PATH + 'test' + '-' + subontology + '.pkl')

    return train_df, valid_df, test_df

def get_data(data_frame):
    print((data_frame['labels'].values.shape))
    data = data_frame['sequences'].values
    labels = (lambda v: np.hstack(v).reshape(len(v), len(v[0])))(data_frame['labels'].values)
    return data, labels

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

def predict(model_path, X_test):
    model = keras.models.load_model(model_path + '.h5', custom_objects={'KMaxPooling': KMaxPooling,'SeqSelfAttention': SeqSelfAttention})
    preds = model.predict(X_test)
    return preds


def compute_performance(preds, labels):
    preds = np.round(preds, 2)
    # f = 0
    # p = 0
    # r = 0
    # a = 0

    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        f = 0.0
        p = 0.0
        r = 0.0
        a = 0.0
        total = 0
        p_total = 0
        a_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            tn = len(labels[i, :])

            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                acc = (1.0 * (tp + tn)) / (1.0 * (tp + fp + tn + fn))
                p += precision
                r += recall
                a += acc
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        a /= a_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
    return f, p, r, a

if __name__ == '__main__':
    main()
