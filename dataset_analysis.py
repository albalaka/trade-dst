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
    dialogues = json.load(open(file))
    for dialogue_dict in tqdm(dialogues):
        # for dialogue_dict in dialogues:
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
                    if word.ent_iob_ == "I":
                        NER[-1] = f"{NER[-1]} {word}"

                # for i, ent in enumerate(NER):
                #     if ent[:5] == 'about':
                #         NER[i] = ent[6:]
                #     elif ent in GENERAL_TYPO.keys():
                #         NER[i] = GENERAL_TYPO[ent]

                # tmp_TP, tmp_samples_TP, tmp_FP, tmp_samples_FP, tmp_FN, tmp_samples_FN = compare_single_turn_GT_NER_labels(GT, NER)
                tmp_TP, tmp_samples_TP, tmp_FP, tmp_samples_FP, tmp_FN, tmp_samples_FN = compare_NER_IN_GT(GT, NER)
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


def compare_GT_NER_labels_BERT(file, BERT_model, tokenizer):
    TP, FP, FN = 0, 0, 0
    samples_TP, samples_FP, samples_FN = [], [], []
    dialogues = json.load(open(file))
    for dialogue_dict in tqdm(dialogues):
        # for dialogue_dict in dialogues:
        skip = False
        # skip police and hospital domains
        for domain in dialogue_dict['domains']:
            if domain not in EXPERIMENT_DOMAINS:
                skip = True

        if not skip:
            for turn in dialogue_dict['dialogue']:
                GT = [v for ds, v in turn['turn_label']]
                res = ner(turn['transcript'])
                VAL = BERT_model.predict_sentence_values(tokenizer, turn['transcript'])

                # for i, ent in enumerate(NER):
                #     if ent[:5] == 'about':
                #         NER[i] = ent[6:]
                #     elif ent in GENERAL_TYPO.keys():
                #         NER[i] = GENERAL_TYPO[ent]

                tmp_TP, tmp_samples_TP, tmp_FP, tmp_samples_FP, tmp_FN, tmp_samples_FN = compare_NER_IN_GT(GT, VAL)
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


def compare_NER_IN_GT(GT, NER):
    TP, FP, FN = 0, 0, 0
    samples_TP, samples_FP, samples_FN = [], [], []
    for pred in NER:
        found = False
        for label in GT:
            if label in pred:
                TP += 1
                samples_TP.append(pred)
                found = True
        if not found:
            FP += 1
            samples_FP.append(pred)

    for label in GT:
        found = False
        for pred in NER:
            if label in pred:
                found = True
        if not found:
            FN += 1
            samples_FN.append(label)

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


def get_slot_information(ontology):
    ontology_domains = dict(
        [(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    slots = [k.replace(" ", "").lower() if ("book" not in k)
             else k.lower() for k in ontology_domains.keys()]
    return slots


def compare_GT_NER_labels_separate_slots(file):
    ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    all_slots = {slot: {"TP": 0,
                        "FP": 0,
                        "FN": 0,
                        "samples_TP": [],
                        "samples_FP": [],
                        "samples_FN": []
                        }
                 for slot in get_slot_information(ontology)
                 }


def get_slot_values(file):
    slots = {}
    total_turns = 0
    empty_turns = 0

    dialogues = json.load(open(file))
    for dialogue_dict in dialogues:
        # for dialogue_dict in dialogues:
        skip = False
        # skip police and hospital domains
        for domain in dialogue_dict['domains']:
            if domain not in EXPERIMENT_DOMAINS:
                skip = True

        if not skip:
            for turn in dialogue_dict['dialogue']:
                if not turn['turn_label']:
                    empty_turns += 1
                for ds, v in turn['turn_label']:
                    # For some reason, a single datum in train set, PMUL2256,
                    #  does not have hospital as a domain, but it has hospital related slots
                    if 'hospital' in ds:
                        continue
                    if ds not in slots.keys():
                        slots[ds] = {v: 1}
                    elif v not in slots[ds]:
                        slots[ds][v] = 1
                    else:
                        slots[ds][v] += 1
                total_turns += 1
    for ds in slots:
        slots[ds] = {k: v for k, v in sorted(slots[ds].items(), key=lambda item: item[1], reverse=True)}
    return slots, total_turns, empty_turns


def view_multiwoz_metadata():
    train_slots, train_total_turns, train_empty_turns = get_slot_values(file_train)
    dev_slots, dev_total_turns, dev_empty_turns = get_slot_values(file_dev)
    test_slots, test_total_turns, test_empty_turns = get_slot_values(file_test)

    combined_slots = {}
    total_slot_values = 0
    for slot_set in [train_slots, dev_slots, test_slots]:
        for ds in slot_set:
            if ds not in combined_slots.keys():
                combined_slots[ds] = {}
            for v in slot_set[ds]:
                if v not in combined_slots[ds].keys():
                    combined_slots[ds][v] = slot_set[ds][v]
                else:
                    combined_slots[ds][v] += slot_set[ds][v]
                total_slot_values += slot_set[ds][v]

    combined_slots = {ds: {k: v for k, v in sorted(combined_slots[ds].items(), key=lambda item: item[1], reverse=True)} for ds in combined_slots}

    print("OVERVIEW".ljust(30), "# train/dev/test")
    print(f"Total turns \t\t\t{train_total_turns} {dev_total_turns} {test_total_turns}")
    print(f"Empty turns \t\t\t{train_empty_turns} {dev_empty_turns} {test_empty_turns}")
    print("\nSamples per slot type:".ljust(30), "# train/dev/test".ljust(20), "# unique values".ljust(20), "Percent of total slots")
    for slot in train_slots:
        tot_single_slot = sum(train_slots[slot].values())+sum(dev_slots[slot].values())+sum(test_slots[slot].values())
        print(
            f"{slot: <30}\t{sum(train_slots[slot].values())}/{sum(dev_slots[slot].values())}/{sum(test_slots[slot].values()): <15}{len(combined_slots[slot]): <15}{tot_single_slot/total_slot_values: .3f}")

    print("\nDETAILS")
    print("Most frequent values per slot".ljust(30), "VALUE".ljust(10), "COUNT".ljust(10), "PERCENT".ljust(10), "PERCENT ALL TURNS")
    for slot in train_slots:
        print(f"\n{slot}")
        for k in list(combined_slots[slot].keys())[:10]:
            print(
                f"{k: >35} {combined_slots[slot][k]: >10} {combined_slots[slot][k]/sum(combined_slots[slot].values()):>10.2f} {combined_slots[slot][k]/(train_total_turns+dev_total_turns+test_total_turns):>10.2f}")

# def GT_in_turn(file):
#     dialogues = json.load(open(file))
#     for dialogue_dict in tqdm(dialogues):
#         # for dialogue_dict in dialogues:
#         skip = False
#         # skip police and hospital domains
#         for domain in dialogue_dict['domains']:
#             if domain not in EXPERIMENT_DOMAINS:
#                 skip = True

#         if not skip:


# TP, samples_TP, FP, samples_FP, FN, samples_FN = compare_GT_NER_labels_separate_slots(file_dev)
# view_multiwoz_metadata()
