import json
import random

import numpy as np
from typing import List, Tuple
import copy
import sys

import torch
import torch.nn as nn

# Transformers
from transformers import AutoTokenizer
from transformers import AutoModel

from model import Model

from collections import defaultdict


def build_model_34(language: str, device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    chk_name_34 = ""
    if language == "EN":
        chk_name_34 = "name__base_model_pos_verbatlas___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__128___pos_dim__128.pt"
    if language == "ES":
        chk_name_34 = "name__base_model_pos_verbatlas_fine_tuning_from_english_chk___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__128___pos_dim__128.pt"
    if language == "FR":
        chk_name_34 = "name__base_model_pos_verbatlas_fine_tuning_from_english_chk___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__128___pos_dim__128.pt"
    return StudentModel(device, language=language, chk_name_34 = chk_name_34)

def build_model_234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    chk_name_34 = ""
    chk_name_2 = ""
    if language == "EN":
        chk_name_34 = "name__base_model_pos_verbatlas___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__128___pos_dim__128.pt"
        chk_name_2 = "name__base_model_en___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__200.pt"
    if language == "ES":
        chk_name_34 = "name__base_model_pos_verbatlas_fine_tuning_from_english_chk___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__128___pos_dim__128.pt"
        chk_name_2 = "name__base_model_es_fine_tuning_from_english_chk___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__200.pt"
    if language == "FR":
        chk_name_34 = "name__base_model_pos_verbatlas_fine_tuning_from_english_chk___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__128___pos_dim__128.pt"
        chk_name_2 = "name__base_model_fr_fine_tuning_from_english_chk___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__200.pt"
    return StudentModel(device, language=language, chk_name_34 = chk_name_34, chk_name_2 = chk_name_2, step = "234")


def build_model_1234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    chk_name_34 = ""
    chk_name_2 = ""
    chk_name_1 = ""
    if language == "EN":
        chk_name_34 = "name__base_model_pos_verbatlas___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__128___pos_dim__128.pt"
        chk_name_2 = "name__base_model_en___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__200.pt"
        chk_name_1 = "name__base_model_en___transformer_name__bert-base-uncased___hidden_dim_LSTM__200.pt"
    if language == "ES":
        chk_name_34 = "name__base_model_pos_verbatlas_fine_tuning_from_english_chk___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__128___pos_dim__128.pt"
        chk_name_2 = "name__base_model_es_fine_tuning_from_english_chk___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__200.pt"
        chk_name_1 = "name__base_model_es___transformer_name__bert-base-uncased___hidden_dim_LSTM__200.pt"
    if language == "FR":
        chk_name_34 = "name__base_model_pos_verbatlas_fine_tuning_from_english_chk___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__128___pos_dim__128.pt"
        chk_name_2 = "name__base_model_fr_fine_tuning_from_english_chk___transformer_name__bert-base-uncased___hidden_dim_LSTM__200___embedding_dim__200.pt"
        chk_name_1 = "name__base_model_fr___transformer_name__bert-base-uncased___hidden_dim_LSTM__200.pt"
    return StudentModel(device, language=language, chk_name_34 = chk_name_34, chk_name_2 = chk_name_2, chk_name_1 = chk_name_1, step = "1234")


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, language: str, return_predicates=False):
        self.language = language
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence["pos_tags"]:
            prob = self.baselines["predicate_identification"].get(pos, dict()).get(
                "positive", 0
            ) / self.baselines["predicate_identification"].get(pos, dict()).get(
                "total", 1
            )
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(
            zip(sentence["lemmas"], predicate_identification)
        ):
            if (
                not is_predicate
                or lemma not in self.baselines["predicate_disambiguation"]
            ):
                predicate_disambiguation.append("_")
            else:
                predicate_disambiguation.append(
                    self.baselines["predicate_disambiguation"][lemma]
                )
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence["dependency_relations"]:
            prob = self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get("positive", 0) / self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get(
                "total", 1
            )
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(
            sentence["dependency_relations"], argument_identification
        ):
            if not is_argument:
                argument_classification.append("_")
            else:
                argument_classification.append(
                    self.baselines["argument_classification"][dependency_relation]
                )

        if self.return_predicates:
            return {
                "predicates": predicate_disambiguation,
                "roles": {i: argument_classification for i in predicate_indices},
            }
        else:
            return {"roles": {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path="data/baselines.json"):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines

class SRL_Model_34(nn.Module):
    def __init__(self, hparams):
        super(SRL_Model_34, self).__init__()
        # Number of classes
        self.num_labels = hparams["num_labels"]
        
        # Embeddings
        self.predicate_embedding = torch.nn.Embedding(3, hparams["embedding_dim"], padding_idx = 0)
        self.pos_embedding = torch.nn.Embedding(hparams["pos_number"], hparams["pos_dim"], padding_idx = 0)

        # Transformer
        self.transformer_model = AutoModel.from_pretrained(hparams["transformer_name"], output_hidden_states=True)
        transformer_output_dim = self.transformer_model.config.hidden_size + hparams["embedding_dim"] + hparams["pos_dim"]
        
        # LSTM
        self.lstm = nn.LSTM(transformer_output_dim,
                            hparams["hidden_dim_LSTM"],
                            bidirectional = hparams["bidirectional"],
                            num_layers = hparams["num_layers_LSTM"], 
                            dropout = 0,
                            batch_first = True)
    
        # If is the LSTM is bidirectional, then the output dimension is duplicated
        lstm_output_dim = hparams["hidden_dim_LSTM"] if hparams["bidirectional"] is False else hparams["hidden_dim_LSTM"] * 2
        
        # Classifier
        self.classifier = nn.Linear(lstm_output_dim, hparams["num_labels"])
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        predicate_idx: torch.Tensor = None,
        pos_idx: torch.Tensor = None
    ) -> torch.Tensor:
        transformer_input = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask
        }
        if token_type_ids != None:
          transformer_input["token_type_ids"] = token_type_ids
        transformers_outputs = self.transformer_model(**transformer_input)
        transformers_outputs_sum = torch.stack(transformers_outputs.hidden_states[-4:], dim=0).sum(dim=0)

        predicate_embedding = self.predicate_embedding(predicate_idx).squeeze(2)

        """
        This "if" is an alternative found at the last minute to solve 
        a problem while still getting almost identical results. 
        Asking if I could use the "pos_tags" field, I was told yes 
        discovering in the end that for steps 1-2, rightly, are not available, 
        and consequently in the "1234" or "234" mode the model that handles 
        steps 3-4 could not handle "pos_tags". 
        Reason why, if I am in mode "34" everything is normal, 
        while in modes "1234" and "234" this trick of duplicate predicate_embedding is used
        """
        if pos_idx is not None:
            pos_embedding = self.pos_embedding(pos_idx).squeeze(2)
            tr_out_with_predicate = torch.cat((transformers_outputs_sum, predicate_embedding, pos_embedding), dim = -1)
        else:
            tr_out_with_predicate = torch.cat((transformers_outputs_sum, predicate_embedding, predicate_embedding), dim = -1)

        o, (h, c) = self.lstm(tr_out_with_predicate)
        logits = self.classifier(o)
        
        output = {"logits": logits}

        predictions = logits.argmax(dim=-1)
        output["predictions"] = predictions
        return output

class SRL_Model_1(nn.Module):
    def __init__(self, hparams):
        super(SRL_Model_1, self).__init__()
        # Number of classes
        self.num_labels = hparams["num_labels"]

        # Transformer
        self.transformer_model = AutoModel.from_pretrained(hparams["transformer_name"], output_hidden_states=True)
        transformer_output_dim = self.transformer_model.config.hidden_size

        # LSTM
        self.lstm = nn.LSTM(transformer_output_dim,
                            hparams["hidden_dim_LSTM"],
                            bidirectional = hparams["bidirectional"],
                            num_layers = hparams["num_layers_LSTM"], 
                            dropout = 0,
                            batch_first = True)
    
        # If is the LSTM is bidirectional, then the output dimension is duplicated
        lstm_output_dim = hparams["hidden_dim_LSTM"] if hparams["bidirectional"] is False else hparams["hidden_dim_LSTM"] * 2
        
        # Classifier
        self.classifier = nn.Linear(lstm_output_dim, hparams["num_labels"])
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None
    ) -> torch.Tensor:
        transformer_input = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask
        }
        if token_type_ids != None:
          transformer_input["token_type_ids"] = token_type_ids
        transformers_outputs = self.transformer_model(**transformer_input)
        transformers_outputs_sum = torch.stack(transformers_outputs.hidden_states[-4:], dim=0).sum(dim=0)

        o, (h, c) = self.lstm(transformers_outputs_sum)
        logits = self.classifier(o)
        
        output = {"logits": logits}
        predictions = logits.argmax(dim=-1)
        output["predictions"] = predictions
        return output

class SRL_Model_2(nn.Module):
    def __init__(self, hparams):
        super(SRL_Model_2, self).__init__()
        # Number of classes
        self.num_labels = hparams["num_labels"]

        # Embeddings
        self.predicate_embedding = torch.nn.Embedding(3, hparams["embedding_dim"], padding_idx = 0)

        # Transformer
        self.transformer_model = AutoModel.from_pretrained(hparams["transformer_name"], output_hidden_states=True)
        transformer_output_dim = self.transformer_model.config.hidden_size + hparams["embedding_dim"]
        
        # LSTM
        self.lstm = nn.LSTM(transformer_output_dim,
                            hparams["hidden_dim_LSTM"],
                            bidirectional = hparams["bidirectional"],
                            num_layers = hparams["num_layers_LSTM"], 
                            dropout = 0,
                            batch_first = True)
    
        # If is the LSTM is bidirectional, then the output dimension is duplicated
        lstm_output_dim = hparams["hidden_dim_LSTM"] if hparams["bidirectional"] is False else hparams["hidden_dim_LSTM"] * 2
        
        # Classifier
        self.classifier = nn.Linear(lstm_output_dim, hparams["num_labels"])
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        predicate_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        transformer_input = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask
        }
        if token_type_ids != None:
          transformer_input["token_type_ids"] = token_type_ids
        transformers_outputs = self.transformer_model(**transformer_input)
        transformers_outputs_sum = torch.stack(transformers_outputs.hidden_states[-4:], dim=0).sum(dim=0)

        embedding = self.predicate_embedding(predicate_idx).squeeze(2)
        tr_out_with_predicate = torch.cat((transformers_outputs_sum, embedding), dim = -1)

        o, (h, c) = self.lstm(tr_out_with_predicate)
        logits = self.classifier(o)

        output = {"logits": logits}
        predictions = logits.argmax(dim=-1)
        output["predictions"] = predictions
        return output

class StudentModel():
    def __init__(
            self, 
            device, 
            language: str, 
            chk_name_34: str, 
            chk_name_1: str = None, 
            chk_name_2: str = None, 
            step: str = "34"
        ):
        """
        Args:
            device: "cuda" or "cpu"
            language: "EN", "ES" or "FR" to load the specific checkpoints
            chk_name_34: name of the checkpoint of the model for the step 3-4
            chk_name_1: name of the checkpoint of the model for the step 1
            chk_name_2: name of the checkpoint of the model for the step 2
            step: the modality of the evaluation: "34", "234" or "1234"
        """
        self.step = step
        self.language = language
        self.device = device

        self.chk_name_34 = chk_name_34
        self.chk_name_1 = chk_name_1
        self.chk_name_2 = chk_name_2

        self.roles = ["_", "agent", "asset", "attribute", "beneficiary", "cause", "co-agent", 
                        "co-patient", "co-theme", "destination", "experiencer", "extent", 
                        "goal", "idiom", "instrument", "location", "material", "patient", 
                        "product", "purpose", "recipient", "result", "source", "stimulus", 
                        "theme", "time", "topic", "value"]
        
        self.pos = ["<PAD>", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", 
                    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
        self.id2label, self.label2id = self.encode_roles(self.roles)
        self.id2pos, self.pos2id = self.encode_pos(self.pos)
        
        self.predicates_label, self.id2predicate_label, self.predicate2id_label = self.encode_predicates()
        self.srl_model_34 = None
        self.srl_model_1 = None
        self.srl_model_2 = None
    
    def encode_predicates(self):
        """
        Method used to read the predicates from VerbAtlas.
        Return:
            predicates_label: list of predicates
            id2predicate_label: map the id to a predicate
            predicate2id_label: map the predicate to an id
        """
        with open("./model/VA_frame_info.tsv", mode='r', encoding='utf-8') as json_file:
            json_list_verbatlas_info = list(json_file)
        id2predicate_label = {0: "_"}
        predicate2id_label = {"_": 0}
        predicates_label = ["_"]
        for id, info in enumerate(json_list_verbatlas_info[1:]):
            info_splitted = info.replace("\n", "").split("\t")
            name = info_splitted[1].upper()
            predicates_label.append(name)
            id2predicate_label[id+1] = name
            predicate2id_label[name] = id+1
        return predicates_label, id2predicate_label, predicate2id_label

    def encode_pos(self, pos):
        """
        Method used to encode the pos_tags.
        Return:
            id2pos: map the id to a pos_tag
            pos2id: map the pos_tag to an id
        """
        id2pos= {}
        pos2id = {}
        for id, label in enumerate(pos):
            id2pos[id] = label
            pos2id[label] = id
        return id2pos, pos2id

    def encode_roles(self, roles):
        """
        Method used to encode the roles.
        Return:
            id2label: map the id to a role
            label2id: map the role to an id
        """
        id2label = {}
        label2id = {}
        for id, label in enumerate(roles):
            id2label[id] = label
            label2id[label] = id
        return id2label, label2id

    def read_from_verbatlas(self):
        """
        Method used to read information from VerbAtlas, 
        specifically the first two ARGS for each predicate, 
        saving everything in dictionaries.
        Return:
            id2predicate: map the id to a predicate.
            predicate2id: map the predicate to an id.
            predicate2args: map a predicate to a list of the first two args
        """
        with open("./model/VA_frame_info.tsv", mode='r', encoding='utf-8') as json_file:
            json_list_verbatlas_info = list(json_file)
        with open("./model/VA_frame_pas.tsv", mode='r', encoding='utf-8') as json_file:
            json_list_verbatlas_pas = list(json_file)
        id2predicate = {}
        predicate2id = {}
        predicate2args = defaultdict(lambda: [])
        for info, pas in zip(json_list_verbatlas_info[1:], json_list_verbatlas_pas[1:]):
            # All info are splitted with a "\t" character
            info_splitted = info.replace("\n", "").split("\t")
            pas_splitted = pas.replace("\n", "").split("\t")
            name = info_splitted[1].upper()
            id = info_splitted[0]
            id2predicate[id] = name
            predicate2id[name] = id
            for arg in pas_splitted[1:3]:
                predicate2args[name].append(arg.lower())
        predicate2args = dict(predicate2args)
        return id2predicate, predicate2id, predicate2args

    def decode_hparams(self, chk_name):
        """
        Method used to get the hparams from a checkpoint name.
        Return:
            hparams: dictionary with the hparams
        """
        hparams = {}
        if chk_name != None:
            for attribute in chk_name.replace(".pt", "").split("___"):
                key = attribute.split("__")[0]
                value = attribute.split("__")[1].replace("%", "/")
                if (key != "transformer_name" and key != "name"):
                    if value == "True":
                        value = True
                    elif  value == "False":
                        value = False
                    elif '.' in value or "1e-" in value:
                        value = float(value)
                    else:
                        value = int(value)
                hparams[key] = value
        return hparams

    def predict(self, sentence):
        """
        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                    {
                        "words":
                            [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                        "lemmas":
                            ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        # NOTE: you do NOT have a "predicates" field here.
                    },

        Returns:
            A dictionary with your predictions:
                - If you are doing argument identification + argument classification:
                    {
                        "roles": list of lists, # A list of roles for each predicate in the sentence.
                    }
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
                        "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence.
                    }
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                        "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence.
                    }
        """
        # For the first sentence, initialize the models
        if (self.srl_model_34 == None):
            chk_name_1 = self.chk_name_1
            chk_name_2 = self.chk_name_2
            chk_name_34 = self.chk_name_34
            # Get the information from each checkpoint name
            hparams_1 = self.decode_hparams(chk_name_1)
            hparams_2 = self.decode_hparams(chk_name_2)
            hparams_34 = self.decode_hparams(chk_name_34)

            # If the modality is "1234"
            if self.step == "1234":
                # The labels are ["_", "predicate"]
                hparams_1["num_labels"] = 2
                hparams_1["num_layers_LSTM"] = 2
                hparams_1["bidirectional"] = True
                # Load and set to eval the model for step 1
                self.srl_model_1 = SRL_Model_1(hparams_1).to(self.device)
                self.srl_model_1.load_state_dict(torch.load("./model/checkpoints_1/" + self.language + "/" + chk_name_1, map_location=torch.device(self.device)))
                self.srl_model_1.eval()
                self.tokenizer_1 = AutoTokenizer.from_pretrained(hparams_1["transformer_name"])

            # If the modality is "1234" or "234"
            if self.step == "234" or self.step == "1234":
                # The labels are the number of type of predicates in VerbAtlas
                hparams_2["num_labels"] = len(self.predicates_label)
                hparams_2["num_layers_LSTM"] = 2
                hparams_2["bidirectional"] = True
                # Load and set to eval the model for step 2
                self.srl_model_2 = SRL_Model_2(hparams_2).to(self.device)
                self.srl_model_2.load_state_dict(torch.load("./model/checkpoints_2/" + self.language + "/" + chk_name_2, map_location=torch.device(self.device)))
                self.srl_model_2.eval()
                self.tokenizer_2 = AutoTokenizer.from_pretrained(hparams_2["transformer_name"])
            
            # If the modality is "1234" or "234" or "34"
            # The label are the number of roles in VerbAtlas
            hparams_34["num_labels"] = len(self.roles)
            hparams_34["pos_number"] = len(self.pos)
            hparams_34["num_layers_LSTM"] = 2
            hparams_34["bidirectional"] = True
            # Load and set to eval the model for step 3-4
            self.srl_model_34 = SRL_Model_34(hparams_34).to(self.device)
            self.srl_model_34.load_state_dict(torch.load("./model/checkpoints_34/" + self.language + "/" + chk_name_34, map_location=torch.device(self.device)))
            self.srl_model_34.eval()
            self.tokenizer_34 = AutoTokenizer.from_pretrained(hparams_34["transformer_name"])
            self.id2predicate, self.predicate2id, self.predicate2args = self.read_from_verbatlas()

            print("Models' hparams")
            print("model hparams_1:", hparams_1)
            print("model hparams_2:", hparams_2)
            print("model hparams_34:", hparams_34)

        output = {}
        # All steps use deep copy of the sentence because of the side effect problem
        # If the modality is "1234"
        if self.step == "1234":
            # Compute the step 1 finding the predicates
            step_1 = self.predict_1(copy.deepcopy(sentence))
            # Compute the step 2 disambiguating the predicates
            step_2 = self.predict_2(copy.deepcopy(sentence), step_1)
            # Compute the step 3-4 finding the arguments and classifying them
            step_34 = self.predict_34(copy.deepcopy(sentence), step_2)
            # The outputs:
            output["predicates"] = step_2
            output["roles"] = step_34
        # If the modality is "234"
        if self.step == "234":
            # Compute the step 2 disambiguating the predicates
            step_2 = self.predict_2(copy.deepcopy(sentence))
            # Compute the step 3-4 finding the arguments and classifying them
            step_34 = self.predict_34(copy.deepcopy(sentence), step_2)
            # The outpus:
            output["predicates"] = step_2
            output["roles"] = step_34
        if self.step == "34":
            # Compute the step 3-4 finding the arguments and classifying them
            step_34 = self.predict_34(copy.deepcopy(sentence))
            # The outputs:
            output["roles"] = step_34
        return output

    def predict_34(self, sentence, step_2 = None):
        """
        Compute the prediction for the step 3. The step_2 is the output of the step 2
        """
        # If it is None, it means the mode is "34" 
        # and therefore we don't have an output from step 2 to use
        if step_2 is not None:
            sentence["predicates"] = step_2
        inputs_model, decode_index = self.create_input_34(sentence)
        empty = True
        output = {}
        # If there is no predicate in the sentence (dataset fault)
        for predicate in sentence["predicates"]:
            if (predicate != "_"):
                empty = False
                break
        if not empty:
            for i, input_model in enumerate(inputs_model):
                # Compute the prediction
                pred = self.srl_model_34(**input_model)
                decoded_pred = []
                previous_idx = None
                predicate_index = 0
                # Reconstruct the prediction using the position 
                # of the original words in the sentence before the tokenizer split.
                for j, idx in enumerate(decode_index[0]):
                    if idx is None:
                        continue
                    elif idx == previous_idx:
                        continue
                    else:
                        if (j == input_model["predicate_idx"][0].tolist().index([2])):
                            predicate_index = idx
                        dec_pred = pred["predictions"][0].tolist()
                        dec_pred = self.id2label[int(dec_pred[j])]
                        decoded_pred.append(dec_pred)
                    previous_idx = idx
                output[predicate_index] = decoded_pred
        return output        


    def predict_1(self, sentence):
        """
        Compute the prediction for the step 1
        """
        input_model, word_ids = self.create_input_1(sentence)
        # Compute the prediction
        predictions = self.srl_model_1(**input_model)["predictions"][0].tolist()
        prediction_post_process = []
        previous_idx = None
        # Reconstruct the prediction using the position 
        # of the original words in the sentence before the tokenizer split.
        for j, idx in enumerate(word_ids):
            if idx is None:
                continue
            elif idx == previous_idx:
                continue
            else:
                prediction_post_process.append(predictions[j])
            previous_idx = idx
        output = []
        # Decode the predictions
        for prediction in prediction_post_process:
            if prediction == 0:
                output.append("_")
            else:
                output.append("predicate")
        return output

    def predict_2(self, sentence, step_1 = None):
        """
        Compute the prediction for the step 2. The step_1 is the output of the step 1
        N.B. The training of this model was done by taking as input 
             sentences, considering only one verb at a time. 
             However, one input considering all predicates together is sufficient 
             to make a prediction, as the model has learned to recognize predicates in sentences
        """
        # If it is None, it means the mode is "234" 
        # and therefore we don't have an output from step 1 to use
        if step_1 is not None:
            sentence["predicates"] = step_1
        else:
            # If is None, sentence["predicates"] have the encoded version. I need to decode first.
            for i in range(len(sentence["predicates"])):
                if int(sentence["predicates"][i]) == 0:
                    sentence["predicates"][i] = "_"
                else:
                    sentence["predicates"][i] = "predicate"
        inputs_model, word_ids = self.create_input_2(sentence)
        # If there is no predicate in the sentence (dataset fault)
        if inputs_model == []:
            return []
        # Compute the prediction
        predictions = self.srl_model_2(**inputs_model)["predictions"][0].tolist()
        prediction_post_process = []
        previous_idx = None
        # Reconstruct the prediction using the position 
        # of the original words in the sentence before the tokenizer split.
        for j, idx in enumerate(word_ids):
            if idx is None:
                continue
            elif idx == previous_idx:
                continue
            else:
                prediction_post_process.append(predictions[j])
            previous_idx = idx
        output = []
        # Decode the prediction
        for prediction in prediction_post_process:
            output.append(self.id2predicate_label[prediction])
        return output

    def create_input_34(self, sentence):
        """
        Method used to create the input for the step 3-4
        N.B. For the problem commented in the model, each "pos_tags" variable is 
             used only if the mode is "34"
        Args:
            sentence: the sentence dictionary to manipulate
        Returns:
            encoded: the inputs for the model
            decode_index: the index of the original sentence 
                          before the split of the tokenizer
        """
        encoded = []
        decode_index = []
        is_first = True
        for i, predicate in enumerate(sentence["predicates"]):
            if predicate == "_":
                continue
            # DATASET ERROR FIX:
            #   A lot of predicates in the dataset end with "-" character 
            if predicate[-1] == "-":
               predicate = predicate[:-1]
            # PREPROCESS
            # Adding "[SEP]" lemma of the predicate and the first two ARGS in VerbAtlas to the sentence
            if not is_first:
                sentence["words"] = sentence["words"][:-4]
                if self.step == "34":    
                    sentence["pos_tags"] = sentence["pos_tags"][:-4]
            is_first = False
            sentence["words"] += ["[SEP]"]
            sentence["words"] += [sentence["lemmas"][i]]
            sentence["words"] += self.predicate2args[predicate]
            # Then add the padding for the pos
            if self.step == "34":
                sentence["pos_tags"] += [self.id2pos[0]]*4
            # Tokenize the sentence
            tokenized = self.tokenizer_34(
                sentence["words"], 
                return_tensors = "pt", 
                truncation = True, 
                padding = True,
                is_split_into_words = True
            )
            tokenized.to(self.device)
            word_ids = tokenized.word_ids()
            previous_word_idx = None
            predicate_idx = []
            if self.step == "34":
                pos_idx = []
            end_sentence = False
            end_word_idx_index = 0
            for j, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None.
                # The predicate and pos is setted to pad
                if word_idx is None:
                    predicate_idx.append(0)
                    if self.step == "34":
                        pos_idx.append(self.pos2id["<PAD>"])
                # For the first token of each word.
                elif word_idx != previous_word_idx:
                    # The sentence is finished 
                    # and the part with additional information begins.
                    if sentence["words"][word_idx] == "[SEP]" and not end_sentence:
                        end_sentence = True
                        end_word_idx_index = j
                    # In this case, we set predicate index to a pad
                    if (end_sentence):
                        predicate_idx.append(0)
                    # In the other case, we set a proper predicate index
                    else:
                        if word_idx == i:
                            predicate_idx.append(2)
                        else:
                            predicate_idx.append(1)
                    # For the pos, there is no problem
                    # since the pad has already been included in the PREPROCESS
                    if self.step == "34":
                        pos_idx.append(self.pos2id[sentence["pos_tags"][word_idx]])
                # For the other tokens in a word id that is None.
                # The predicate and pos is setted to pad
                else:
                    predicate_idx.append(0)
                    if self.step == "34":
                        pos_idx.append(self.pos2id["<PAD>"])
                previous_word_idx = word_idx   
            # Since we need the index of the word in the sentence after the prediction 
            # to reconstruct the original shape, we have to set at None
            # the value of the indexes in the VerbAtlas thinks position
            for k, word_idx in enumerate(word_ids):
                if k >= end_word_idx_index:
                    word_ids[k] = None
            decode_index.append(word_ids)
            predicate_idx = torch.LongTensor(predicate_idx)
            predicate_idx = predicate_idx.unsqueeze(dim = -1).unsqueeze(dim = 0).to(self.device)
            if self.step == "34":
                pos_idx = torch.LongTensor(pos_idx)
                pos_idx = pos_idx.unsqueeze(dim = -1).unsqueeze(dim = 0).to(self.device)
            entry = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "token_type_ids": tokenized["token_type_ids"],
                "predicate_idx": predicate_idx,
            }
            if self.step == "34":
                entry["pos_idx"] = pos_idx
            encoded.append(entry)
        return encoded, decode_index


    def create_input_1(self, sentence):
        """
        Method used to create the input for the step 1
        Args:
            sentence: the sentence dictionary to manipulate
        Returns:
            entry: the inputs for the model
            word_ids: the index of the original sentence 
                          before the split of the tokenizer 
        """
        # Tokenize the sentence
        tokenized = self.tokenizer_1(
                sentence["words"], 
                return_tensors = "pt", 
                truncation = True, 
                padding = True,
                is_split_into_words = True
            )
        tokenized.to(self.device)
        entry = {}
        for key, value in tokenized.items():
            entry[key] = value
        word_ids = tokenized.word_ids()
        return entry, word_ids

    def create_input_2(self, sentence):
        """
        Method used to create the input for the step 2
        Args:
            sentence: the sentence dictionary to manipulate
        Returns:
            entry: the inputs for the model
            word_ids: the index of the original sentence 
                          before the split of the tokenizer 
        """
        # If there is no predicate in the sentence (dataset fault)
        no_predicate = True
        for predicate in sentence["predicates"]:
            if predicate != "_":
                no_predicate = False
        if no_predicate:
            return [], []
        # Tokenize the sentence
        tokenized = self.tokenizer_2(
            sentence["words"], 
            return_tensors = "pt", 
            truncation = True, 
            padding = True,
            is_split_into_words = True
        )
        tokenized.to(self.device)
        entry = {}
        for key, value in tokenized.items():
            entry[key] = value
        word_ids = tokenized.word_ids()
        previous_word_idx = None
        predicate_idx = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None.
            # The predicate is setted to pad
            if word_idx is None:
                predicate_idx.append(0)
            # For the first token of each word.
            elif word_idx != previous_word_idx:
                # Set a proper predicate index
                if sentence["predicates"][word_idx] != self.id2label[0]:
                    predicate_idx.append(2)
                else:
                    predicate_idx.append(1)
            # For the other tokens in a word id that is None.
            # We set the predicate to pad
            else:
                predicate_idx.append(0)
            previous_word_idx = word_idx
        predicate_idx = torch.LongTensor(predicate_idx)
        predicate_idx = predicate_idx.unsqueeze(dim = -1).unsqueeze(dim = 0).to(self.device)
        entry["predicate_idx"] =  predicate_idx
        return entry, word_ids