import numpy as np
import json

import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torchcrf import CRF

from typing import List
from collections import defaultdict

from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    student_model = None

    # If I use Conditional Random Field or not
    IS_CRF = True
    
    # Load the vocabulary with all structure and attributes
    if (IS_CRF):
        with open("./model/vocabularyCRF.json", 'r') as f:
            vocab = json.load(f)
    else:
        with open("./model/vocabulary.json", 'r') as f:
            vocab = json.load(f)

    # Hyperparameters of the model. The dropout is put for illustration purposes as the model is put into eval() with no grad
    hparams = type('',(object,),{"vocab_size": len(vocab["word2id"]),
                                 "hidden_dim": 128,
                                 "word_embedding_dim": 200,
                                 "num_classes": len(vocab["classes"]) + 1,
                                 "bidirectional": True,
                                 "num_layers": 2,
                                 "dropout": 0.5,
                                 "word_embeddings": None,
                                 "device": device,
                                 "vocab": vocab,
                                 "IS_CRF": IS_CRF})()

    # Initialize the model
    my_model = My_Model(hparams).to(device)
    # Load the checkpoint
    if (IS_CRF):
        my_model.load_state_dict(torch.load('./model/chkCRF.pt', map_location=torch.device(device)))
    else:
        my_model.load_state_dict(torch.load('./model/chk.pt', map_location=torch.device(device)))
    # Freeze the layer and set dropout to 0
    my_model.eval()

    # Create and return the Model
    student_model = StudentModel(my_model)
    if (student_model == None):
        return RandomBaseline()
    else:
        return student_model


class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class My_Model(nn.Module):
    '''
        Architecture on which I trained.
        P.S. For this class I have taken inspiration from the Notebook 6 shown in the lecture
    '''
    def __init__(self, hparams):
        """
            Args:
                hparams: all the parameters that we need in the model.
            P.S. For this method I have taken inspiration from the Notebook 6 shown in the lecture
        """
        super(My_Model, self).__init__()
        self.vocab = hparams.vocab
        self.device = hparams.device
        # Whether I use CRF or not
        self.IS_CRF = hparams.IS_CRF

        # Random word embedding
        self.word_embedding = nn.Embedding(hparams.vocab_size, 
                                        hparams.word_embedding_dim, 
                                        padding_idx=self.vocab["word2id"][self.vocab["PAD_TOKEN"]])
        # Pretrained word embedding
        if hparams.word_embeddings is not None:
            self.word_embedding.weight.data.copy_(hparams.word_embeddings)
        
        # LSTM
        self.lstm = nn.LSTM(hparams.word_embedding_dim,
                            hparams.hidden_dim, 
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0,
                            batch_first = True)
        
        # If is the LSTM is bidirectional, then the output dimension is duplicated
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        
        # Classifier
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)
        
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(hparams.dropout)

        if (self.IS_CRF):
            # Conditional Random Field layer
            self.crf = CRF(hparams.num_classes, batch_first=True)

            # Log softmax (because the crf level returns a log likelihood 
            # unlike the cross entropy loss which is a combination 
            # of log likelihood and log softmax)
            self.softmax = nn.LogSoftmax(-1)
  
    def forward(self, x):
        '''
            Forward of the model:
                Input: (batch_size, window_size)
                Output: (batch_size, window_size, num_classes)
            P.S. For this method I have taken inspiration from the Notebook 6 shown in the lecture
        '''
        # (batch_size, window_size) 
        # The last dim is replaced with index in word embedding
        embeddings = self.word_embedding(x)
        # (batch_size, window_size, embedding_dim)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        # if LSTM  (batch_size, window_size, hidden_dim)
        # if BLSTM (batch_size, window_size, hidden_dim * 2)
        o = self.dropout(o)
        output = self.classifier(o)
        # (batch_size, window_size, num_classes)
        return output

    def predict_crf(self, input_tensor):
        '''
            With input input_tensor;
            Return the crf decoded predictions.
        '''
        # Forward
        emissions = self.__call__(input_tensor)
        if (self.IS_CRF):
            # Decode
            predictions = self.crf.decode(emissions)
        return predictions

class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, model):
        """
            Args:
                model: my model.
        """
        self.model = model
        self.vocab = self.model.vocab
        self.device = self.model.device
        # Whether I use CRF or not
        self.IS_CRF = self.model.IS_CRF

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        """
            Predicting a set of inputs
            Parameter:
                tokens: A list of sentences where a sentence is a list of words.
            Return:
                outputs: A list of sentence predictions where a sentence prediction is a list of label.
        """
        self.model.eval()
        with torch.no_grad():
            # Create the input windows
            inputs = Inputs(self.vocab, tokens, device = self.device)
            # List of outputs
            outputs = []

            # For each sentence
            for se_idx, sentence in enumerate(inputs):
                # Sentence len... useful for reconstructing the sentence later
                sentence_len = len(tokens[se_idx])
                # This index is the starting point of the window in the sentence
                index = 0
                # Dictionary where the key is the index in the sentence 
                # and the value a list of prediction for the word at this index
                word_predictions = defaultdict(lambda : [])

                # For each window in the sentence
                for window in sentence:
                    # Input
                    batch1 = window.unsqueeze(0)
                    # Prediction
                    if (self.IS_CRF):
                        prediction = self.model.predict_crf(batch1)[0]
                    else:
                        prediction = self.model(batch1).squeeze()

                    # For each word prediction in the window
                    for window_index, predicted_label in enumerate(prediction):
                        # At index "Starting point" plus index in the window, add the prediction
                        word_predictions[window_index + index].append(predicted_label)
                    # Update the starting point because the next window is shifted
                    index += inputs.window_shift
                    
                # Create the final prediction based on the list of prediction for each words
                if (self.IS_CRF):
                    sentence_prediction = np.zeros(max(list(word_predictions.keys())) + 1)
                else:
                    sentence_prediction = np.zeros((max(list(word_predictions.keys())) + 1, len(self.vocab["classes"]) + 1))
                # For each word
                for key, value in word_predictions.items():
                    if (self.IS_CRF):
                        # If I use CRF take the prediction with majority criteria
                        sentence_prediction[key] = max(set(value), key = value.count)
                    else:
                        # Whitout CRF take the mean prediction
                        sentence_prediction[key] = torch.mean(torch.stack(value), 0).cpu().detach().numpy()
                    
                # Decode the predictions with id to label
                sentence_prediction_decoded = inputs.decode_output(sentence_prediction, self.IS_CRF)
                # if (self.IS_CRF):
                #     for pred in sentence_prediction:
                #         label = self.vocab["id2label"][str(int(pred))]
                #         if (label == self.vocab["PAD_TOKEN"]):
                #             label = "O"
                #         sentence_prediction_decoded.append(label)
                # else:
                #     for pred in sentence_prediction:
                #         sentence_prediction_decoded.append(inputs.decode_output(torch.tensor(pred).unsqueeze(0))[0])

                # Reconstruct the sentence by discarding the padding at the end.
                sentence_prediction_decoded = sentence_prediction_decoded[:sentence_len]
                outputs.append(sentence_prediction_decoded)

            # Post process:
            #   If a word is predicted as I-CLASS_i
            #   And if there isnt a prediction B-CLASS_i before all the I-CLASS_i
            #   Then the first I-CLASS_i in the series is setted at B-CLASS_i
            for i in range(len(outputs)):
                for j in range(len(outputs[i])):
                    if (outputs[i][j][0] == "I"):
                        if (j == 0):
                            outputs[i][j] = "B" + outputs[i][j][1:]
                        index = j
                        l = outputs[i][index]
                        while (outputs[i][index] == l and index > 0):
                            index -= 1
                        if (outputs[i][index] != "B" + l[1:]):
                            outputs[i][index+1] = "B" + l[1:]
            
            # Print for each sentence and for each word the prediction
            for i, token in enumerate(tokens):
                print("sentence", i)
                for j, word in enumerate(token):
                    print("\tword " + str(j) + ":", word, "\t|\t", "prediction:", outputs[i][j]) #0.6730
            return outputs

class Inputs(Dataset):
    '''
        Class that takes care of creating the dataset, 
        creating tensors that can be used with the model, 
        representing windows of fixed length, 
        which are filled with the words of the sentence plus, if necessary, the padding.
        P.S. For this class I have taken inspiration from the Notebook 6 shown in the lecture
    '''
    def __init__(self, vocab, sentences, device = "cuda"):
        """
            Args:
                vocab: the vocabulary.
                sentences: list of sentences.
                device: "cuda" or "cpu".
            P.S. For this method I have taken inspiration from the Notebook 6 shown in the lecture
        """
        # Vocabulary
        self.vocabulary = vocab

        # Device "cuda" or "cpu"
        self.device = device

        # Window size and shift
        self.window_size = self.vocabulary["window_size"]
        self.window_shift = self.vocabulary["window_step"]

        # Sentences from the set
        self.sentences = sentences

        # Create the windows
        self.data = self.create_windows(self.sentences)

        # Create the data in the dataset
        self.index_dataset()

    def create_windows(self, sentences):
        '''
            Creates fixed-length windows that cover the entire sentence. 
            If the last window exceeds the end of the sentence, 
            then the sentence is filled with padding represented by "None".
            Parameter: 
                - sentences: sentences given by the vocabulary.
            Return: 
                - data: list of windows covering all sentences.
            P.S. For this method I have taken inspiration from the Notebook 6 shown in the lecture
        '''
        data = []
        # For each sentence
        for sentence in sentences:
            data_sentence = []
            # Create the window slice
            for i in range(0, len(sentence), self.window_shift):
                window = sentence[i:i+self.window_size]
                # If is the last window, fill with None and continue to the next sentence
                if len(window) < self.window_size:
                    window = window + [None]*(self.window_size - len(window))
                    data_sentence.append(window)
                    break
                data_sentence.append(window)
            data.append(data_sentence)
        return data

    def index_dataset(self):
        '''
            Creates a dataset entry which is a list of inputs representing 
            the input tensor window to be given to the model.
            P.S. For this method I have taken inspiration from the Notebook 6 shown in the lecture
        '''
        self.encoded_data = list()
        # for each sentence
        for i in range(len(self.data)):
            encoded_sentence = list()
            # for each window
            for j in range(len(self.data[i])):
                elem = self.data[i][j]
                # Translate the words in ids
                encoded_elem = torch.LongTensor(self.encode_text(elem)).to(self.device)
                # Append the new entry
                encoded_sentence.append(encoded_elem)
            # Append the encoded tensor window
            self.encoded_data.append(encoded_sentence)

    def encode_text(self, sentence):
        '''
            Turns a sentence made of text into a sentence made of ids.
            Parameter:
                - sentence: a sentence made of text
            Return:
                - ids: a sentence made of ids
            P.S. For this method I have taken inspiration from the Notebook 6 shown in the lecture
        '''
        ids = []
        # For each entry in the sentence
        for entry in sentence:
            # If is None is a padding
            if (entry is None):
                ids.append(self.vocabulary["word2id"][self.vocabulary["PAD_TOKEN"]])
            # If present in the vocabulary set its id
            elif (self.vocabulary["word2id"].get(entry) != None):
                ids.append(self.vocabulary["word2id"][entry])
            # Else is an unknown
            else:
                ids.append(self.vocabulary["word2id"][self.vocabulary["UNK_TOKEN"]])
        return ids

    def decode_output(self, output, IS_CRF):
        '''
            Transforms a series of label indices into their textual value.
            Parameter:
                - outputs: predictions made by a model in the form of ids
            Return:
                - predictions_decoded: predictions made by a model translated into text
            P.S. For this method I have taken inspiration from the Notebook 6 shown in the lecture
        '''
        predictions_decoded = []
        for pred in output:
            if (IS_CRF):
                label = self.vocabulary["id2label"][str(int(pred))]
                if (label == self.vocabulary["PAD_TOKEN"]):
                    label = "O"
                predictions_decoded.append(label)
            else:
                indices = torch.argmax(torch.tensor(pred).unsqueeze(0), -1).tolist()
                predictions = []
                for i in indices:
                    label = self.vocabulary["id2label"][str(i)]
                    if (label == self.vocabulary["PAD_TOKEN"]):
                        label = "O"
                    predictions.append(label)
                predictions_decoded.append(predictions[0])
        return predictions_decoded

    def __getitem__(self, idx):
        '''
            Return:
                - i-th element of the dataset to iterate on it
            P.S. For this method I have taken inspiration from the Notebook 6 shown in the lecture
        '''
        return self.encoded_data[idx]
