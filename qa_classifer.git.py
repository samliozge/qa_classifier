import builtins
from colorama import Fore,Back
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten,Lambda,RepeatVector,Merge,merge,GRU,Conv1D,GlobalMaxPooling1D,SimpleRNN
from keras.layers import MaxPooling1D, Embedding,LSTM,Bidirectional,Dropout,TimeDistributed,Permute,Activation,Merge
from keras.models import Model
from gensim.models.word2vec import Word2Vec
from keras import backend as K
from sklearn import metrics
from keras.callbacks import EarlyStopping
import random
import tensorflow as tf


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


def recall(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_callback(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

error_to_catch = getattr(builtins,'FileNotFoundError', IOError)

class QADataPreparer:

    def split_data_for_qa_data(self,line):

        print(line)

        line_separtion_index = line.find(" ")

        line_splitted = []

        line_splitted.append(line[:line_separtion_index])
        line_splitted.append(line[line_separtion_index:])


        label_full_part = line_splitted[0]

        label = label_full_part[label_full_part.index(":")+1:]
        print(label)

        sentence = line_splitted[1][1:]

        return {"label":label,"sentence":sentence}

    def __init__(self,fileName):

        self.maxlengthofsentence = 37

        self.content = None
        self.fileData = {}

        self.sentences = []
        self.labels = []
        self.labels2 = []

        self.test_sentences = []
        self.test_labels = []



        try:
            self.content  = open(fileName,encoding="utf-8").read().split("\n")

            random.shuffle(self.content)


            for line in self.content[:len(self.content)-500]:

                line_separtion_index = line.find(" ")


                line_splitted =  []

                line_splitted.append(line[:line_separtion_index])
                line_splitted.append(line[line_separtion_index:])




                if len(line_splitted)>1 and ":" in line_splitted[0]:



                    label_full_part = line_splitted[0]

                    first_part = label_full_part[:label_full_part.index(":")]
                    second_part = label_full_part[label_full_part.index(":")+1:]

                    self.labels.append(second_part)


                    self.sentences.append(line_splitted[1][1:])

                    splitted_word = line_splitted[1][1:].split(" ")
                    if len(splitted_word) > self.maxlengthofsentence:
                        self.maxlengthofsentence = len(splitted_word)

            for sentence in self.content[len(self.content) - 500:]:

                if ":" in sentence:
                    sentence_form = self.split_data_for_qa_data(sentence)
                    self.test_sentences.append(sentence_form["sentence"])
                    self.test_labels.append(sentence_form["label"])


        except error_to_catch:
            print(Fore.RED + "File does not exist")


class NNDataPreparer:

    def __init__(self,qadatapreparer
                 ,maxlen=13
                 ,embedding_model=None
                 ):


        self.tokenizer = Tokenizer()

        self.tokenizer.fit_on_texts(qadatapreparer.sentences)

        self.tokenizer.fit_on_texts(qadatapreparer.test_sentences)

        self.sequences = self.tokenizer.texts_to_sequences(qadatapreparer.sentences)

        self.sequences2 = self.tokenizer.texts_to_sequences(qadatapreparer.test_sentences)



        self.embedding_model = embedding_model

        self.data = pad_sequences(self.sequences,maxlen=qadatapreparer.maxlengthofsentence)

        self.data2 = pad_sequences(self.sequences2,maxlen=qadatapreparer.maxlengthofsentence)

        all_sentences = qadatapreparer.sentences + qadatapreparer.test_sentences



        self.label_number_versions = {}

        for label in qadatapreparer.labels:
            if label not in self.label_number_versions:
                self.label_number_versions[label] = len(self.label_number_versions)+1


        self.labels_in_number_form = list(map(lambda label:self.label_number_versions[label],qadatapreparer.labels))

        self.labels_in_number_form2 = list(map(lambda label:self.label_number_versions[label],qadatapreparer.test_labels))



        self.labels = to_categorical(self.labels_in_number_form)

        self.label2 = to_categorical(self.labels_in_number_form2)

        if embedding_model == None:
            self.embedding_model = Word2Vec([s.lower().split() for s in all_sentences], size=50, window=4, min_count=0, workers=4)
            #self.embedding_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

        self.embedding_matrix  = np.zeros((len(self.embedding_model.wv.vocab) + 1, 50))

        i = 0
        for word in self.embedding_model.wv.vocab:
            self.embedding_matrix[i] = self.embedding_model[word]
            i = i + 1


class RNNModels:

    def __init__(self,embedding_model,
                 embedding_matrix,
                 embedding_length,
                 number_of_classes,
                 input_length,
                 model_type="GRU",
                 epoch=20
                 ):

        print(input_length)

        self.model_type = model_type


        self.embedding_layer = Embedding(len(embedding_model.wv.vocab) + 1,
                                    embedding_length,
                                    weights=[embedding_matrix],
                                    input_length=input_length,
                                    trainable=True)

        self.sequence_input = Input(shape=(input_length,), dtype='int32')

        self.embedded_sequences = self.embedding_layer(self.sequence_input)

        # x = Dense(150)(embedded_sequences)

        # x = Bidirectional(LSTM(32,return_sequences=True))(embedded_sequences)

        self.x = None

        if model_type == "RNN":
            self.x = SimpleRNN(32)(self.embedded_sequences)
        if model_type == "GRU":
            self.x = GRU(32)(self.embedded_sequences)
        elif model_type == "LSTM":
            self.x = LSTM(64)(self.embedded_sequences)
        elif model_type == "CNN":
            #self.x = Dense(32)(self.embedded_sequences)
            self.x = Conv1D(64 ,input_length)(self.embedded_sequences)
            #self.x = Dropout(0.5)(self.x)
            self.x  = GlobalMaxPooling1D()(self.x)
        #self.x = Dense(512,activation='relu')(self.x)


        self.preds = Dense(number_of_classes, activation='softmax')(self.x)

        self.model = Model(self.sequence_input, self.preds)
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy','mse',f1,recall,precision_callback]
                      )

    def train(self,data,labels,batch_size=50,epoch=60):

        earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')


        self.model.fit(data, labels,
                  batch_size=batch_size,
                  epochs=epoch,
                  callbacks=[earlyStopping],
                       validation_split=0.1,
                  )
    def test_accuracy(self,data,labels):
        results = self.model.predict(data,batch_size=128)
        real_results = list(map(lambda x: np.argmax(x), results))
        print(labels)
        print(real_results)
        my_accuracy_score = metrics.accuracy_score(labels, real_results)
        my_f1_score = metrics.f1_score(labels, real_results, average='macro')
        my_precision_score = metrics.precision_score(labels, real_results, average='macro')
        my_recall_score = metrics.recall_score(labels, real_results, average='macro')

        print(self.model_type + " Presicion Score:" + str(my_precision_score))
        print(self.model_type + " Recall Score:" + str(my_recall_score))
        print(self.model_type + " F1 Score:" + str(my_f1_score))
        print(self.model_type + " Accuracy:" + str(my_accuracy_score))







my_data = QADataPreparer("train_5500.label.txt")

my_data_preparer = NNDataPreparer(my_data)

print("")
print(len(my_data.test_labels),len(my_data.test_sentences))
#print(my_data_preparer.labels_in_number_form2)
#print(my_data_preparer.labels_in_number_form2)

my_lstm_network =  RNNModels(embedding_model=my_data_preparer.embedding_model
                             ,embedding_matrix=my_data_preparer.embedding_matrix
                             ,embedding_length=50,
                             model_type="LSTM",
                             input_length=my_data.maxlengthofsentence,
                             number_of_classes=len(my_data_preparer.label_number_versions)+1,

                             )
my_gru_network =  RNNModels(embedding_model=my_data_preparer.embedding_model
                             ,embedding_matrix=my_data_preparer.embedding_matrix
                             ,embedding_length=50,
                             model_type="GRU",
                            input_length=my_data.maxlengthofsentence,
                             number_of_classes=len(my_data_preparer.label_number_versions)+1,

                             )


my_cnn_network =  RNNModels(embedding_model=my_data_preparer.embedding_model
                             ,embedding_matrix=my_data_preparer.embedding_matrix
                             ,embedding_length=50,
                             model_type="CNN",
                             number_of_classes=len(my_data_preparer.label_number_versions)+1,
                            input_length=my_data.maxlengthofsentence)

my_rnn_network = RNNModels(embedding_model=my_data_preparer.embedding_model
                             ,embedding_matrix=my_data_preparer.embedding_matrix
                             ,embedding_length=50,
                             model_type="RNN",
                             number_of_classes=len(my_data_preparer.label_number_versions)+1,
                            input_length=my_data.maxlengthofsentence)


my_lstm_network.train(data=my_data_preparer.data
                      ,labels=my_data_preparer.labels
                      ,epoch=22,
                      batch_size=128,
                      )

my_rnn_network.train(data=my_data_preparer.data
                      ,labels=my_data_preparer.labels
                      ,epoch=22,
                      batch_size=128,
                      )

my_gru_network.train(data=my_data_preparer.data
                      ,labels=my_data_preparer.labels,
                     batch_size=128,
                     epoch=22)


my_cnn_network.train(data=my_data_preparer.data
                      ,labels=my_data_preparer.labels
                      ,epoch=22,
                      batch_size=175,
                      )



print(my_data.maxlengthofsentence)




my_cnn_network.test_accuracy(data=my_data_preparer.data2,labels=my_data_preparer.labels_in_number_form2)
print()

my_lstm_network.test_accuracy(data=my_data_preparer.data,labels=my_data_preparer.labels_in_number_form)
print()

my_gru_network.test_accuracy(data=my_data_preparer.data,labels=my_data_preparer.labels_in_number_form)
print()
my_rnn_network.test_accuracy(data=my_data_preparer.data,labels=my_data_preparer.labels_in_number_form)
