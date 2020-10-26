import json
import os
from tqdm import tqdm

import utils.utils

import en_core_web_sm
ner = en_core_web_sm.load()

kwargs = utils.utils.parse_args()

file_train = 'data/train_dials.json'
file_dev = 'data/dev_dials.json'
file_test = 'data/test_dials.json'

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
GENERAL_TYPO = {
    # type
    "guesthouse": "guest house", "guesthouses": "guest house", "guest": "guest house", "mutiple sports": "multiple sports",
    "sports": "multiple sports", "mutliple sports": "multiple sports", "swimmingpool": "swimming pool", "concerthall": "concert hall",
    "concert": "concert hall", "pool": "swimming pool", "night club": "nightclub", "mus": "museum", "ol": "architecture",
    "colleges": "college", "coll": "college", "architectural": "architecture", "musuem": "museum", "churches": "church",
    # area
    "center": "centre", "center of town": "centre", "near city center": "centre", "in the north": "north", "cen": "centre", "east side": "east",
    "east area": "east", "west part of town": "west", "ce": "centre",  "town center": "centre", "centre of cambridge": "centre",
    "city center": "centre", "the south": "south", "scentre": "centre", "town centre": "centre", "in town": "centre", "north part of town": "north",
    "centre of town": "centre", "cb30aq": "none",
    # price
    "mode": "moderate", "moderate -ly": "moderate", "mo": "moderate",
    # day
    "next friday": "friday", "monda": "monday",
    # parking
    "free parking": "free",
    # internet
    "free internet": "yes",
    # star
    "4 star": "4", "4 stars": "4", "0 star rarting": "none",
    # others
    "y": "yes", "any": "dontcare", "n": "no", "does not care": "dontcare", "not men": "none", "not": "none", "not mentioned": "none",
    '': "none", "not mendtioned": "none", "3 .": "3", "does not": "no", "fun": "none", "art": "none",
}


def compare_GT_NER_labels(file):
    TP, FP, FN = 0, 0, 0
    samples_TP, samples_FP, samples_FN = [], [], []
    dialogues_train = json.load(open(file))
    for dialogue_dict in tqdm(dialogues_train):
        # for dialogue_dict in dialogues_train:
        skip = False
        # skip police and hospital domains
        for domain in dialogue_dict['domains']:
            if domain not in EXPERIMENT_DOMAINS:
                skip = True

        if not skip:
            for turn in dialogue_dict['dialogue']:
                GT = [v for ds, v in turn['turn_label']]
                res = ner(turn['transcript'])
                NER = []
                for word in res:
                    if word.ent_iob_ == "B":
                        NER.append(str(word))
                    if word.ent_iob_ == "I" and str(word) not in ['night', 'nights', 'day', 'days']:
                        NER[-1] = f"{NER[-1]} {word}"

                for i, ent in enumerate(NER):
                    if ent[:5] == 'about':
                        NER[i] = ent[6:]
                    elif ent in GENERAL_TYPO.keys():
                        NER[i] = GENERAL_TYPO[ent]

                tmp_TP, tmp_samples_TP, tmp_FP, tmp_samples_FP, tmp_FN, tmp_samples_FN = compare_single_turn_GT_NER_labels(GT, NER)
                TP += tmp_TP
                samples_TP.extend(tmp_samples_TP)
                FP += tmp_FP
                samples_FP.extend(tmp_samples_FP)
                FN += tmp_FN
                samples_FN.extend(tmp_samples_FN)

    print(f"TP: {TP} FN: {FN} FP: {FP}")
    print("recall: {}".format(TP/(TP+FN)))
    print("precision: {}".format(TP/(TP+FP)))
    return TP, samples_TP, FP, samples_FP, FN, samples_FN


def compare_single_turn_GT_NER_labels(GT, NER):
    TP, FP = 0, 0
    samples_TP, samples_FP, samples_FN = [], [], []
    for pred in NER:
        if pred in GT:
            TP += 1
            samples_TP.append(pred)
        else:
            FP += 1
            samples_FP.append(pred)

    for pred in GT:
        if pred not in NER:
            samples_FN.append(pred)
    FN = len(GT)-TP
    return TP, samples_TP, FP, samples_FP, FN, samples_FN

# def GT_in_