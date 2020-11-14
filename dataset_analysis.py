import json
import os
import re
from tqdm import tqdm

import en_core_web_sm
ner = en_core_web_sm.load()

file_train = 'data/train_dials.json'
file_dev = 'data/dev_dials.json'
file_test = 'data/test_dials.json'

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def fix_general_label_error(labels, slots, drop_slots):
    label_dict = dict([(l["slots"][0][0], l["slots"][0][1]) for l in labels if l["slots"][0][0] not in drop_slots])

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

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])

            # miss match slot and value
            if slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
                slot == "hotel-internet" and label_dict[slot] == "4" or \
                slot == "hotel-pricerange" and label_dict[slot] == "2" or \
                slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
                "area" in slot and label_dict[slot] in ["moderate"] or \
                    "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no":
                    label_dict[slot] = "north"
                elif label_dict[slot] == "we":
                    label_dict[slot] = "west"
                elif label_dict[slot] == "cent":
                    label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we":
                    label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no":
                    label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
                    slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum", "same area as hotel"]:
                label_dict[slot] = "none"

    return label_dict


def load_multiwoz_database(database_file="edited_ontology.json"):
    # Returns all possible values from the database, as a dict
    ontology = json.load(open(database_file))
    return ontology


def find_database_value_in_utterance(utterance, database):
    # either regex search or for loop to find if anything from database is in the utterance

    # slots_train, _, _ = get_slot_values(file_train)
    # slots_dev, _, _ = get_slot_values(file_dev)
    # slots_test, _, _ = get_slot_values(file_test)

    # database = {}
    # for domain in slots_train.keys():
    #     database[domain] = list({**slots_train[domain], **slots_dev[domain], **slots_test[domain]}.keys())

    found_values = {}
    for domain_slot, values in database.items():
        # for val in values:
        #     if val in utterance:
        #         if domain_slot not in found_values.keys():
        #             found_values[domain_slot] = []
        #         found_values[domain_slot].append(val)

        matches = re.findall(r"(?=("+'|'.join(values)+r"))", utterance)
        if matches:
            found_values[domain_slot] = matches
    return found_values


def gather_sort_samples(samples):
    tmp = {}
    for s in samples:
        if s not in tmp.keys():
            tmp[s] = 0
        tmp[s] += 1

    sorted_samples = {k: v for k, v in sorted(tmp.items(), key=lambda x: x[1], reverse=True)}
    return sorted_samples


def compare_GT_NER_labels(file, include_sys=False):
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
                usr_res = ner(turn['transcript'])
                results = [usr_res]
                if include_sys:
                    sys_res = ner(turn['system_transcript'])
                    results.append(sys_res)

                NER = []
                for res in results:
                    for word in res:
                        if word.ent_iob_ == "B":
                            NER.append(str(word))
                        if word.ent_iob_ == "I":
                            NER[-1] = f"{NER[-1]} {word}"

                # tmp_TP, tmp_samples_TP, tmp_FP, tmp_samples_FP, tmp_FN, tmp_samples_FN = compare_single_turn_GT_NER_labels(GT, NER)
                tmp_TP, tmp_samples_TP, tmp_FP, tmp_samples_FP, tmp_FN, tmp_samples_FN = compare_NER_IN_GT(GT, NER)
                TP += tmp_TP
                samples_TP.extend(tmp_samples_TP)
                FP += tmp_FP
                samples_FP.extend(tmp_samples_FP)
                FN += tmp_FN
                samples_FN.extend(tmp_samples_FN)

    samples_TP = gather_sort_samples(samples_TP)
    samples_FP = gather_sort_samples(samples_FP)
    samples_FN = gather_sort_samples(samples_FN)

    print(f"TP: {TP} FN: {FN} FP: {FP}")
    print("recall: {}".format(TP/(TP+FN)))
    print("precision: {}".format(TP/(TP+FP)))
    return TP, samples_TP, FP, samples_FP, FN, samples_FN


def compare_GT_BERT_VE(file, BERT_model, tokenizer, include_sys=False):
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
                VAL = BERT_model.predict_sentence_values(tokenizer, turn['transcript'])
                if include_sys:
                    VAL.extend(BERT_model.predict_sentence_values(tokenizer, turn['system_transcript']))

                tmp_TP, tmp_samples_TP, tmp_FP, tmp_samples_FP, tmp_FN, tmp_samples_FN = compare_NER_IN_GT(GT, VAL)
                TP += tmp_TP
                samples_TP.extend(tmp_samples_TP)
                FP += tmp_FP
                samples_FP.extend(tmp_samples_FP)
                FN += tmp_FN
                samples_FN.extend(tmp_samples_FN)

    samples_TP = gather_sort_samples(samples_TP)
    samples_FP = gather_sort_samples(samples_FP)
    samples_FN = gather_sort_samples(samples_FN)

    print(f"TP: {TP} FN: {FN} FP: {FP}")
    print("recall: {}".format(TP/(TP+FN)))
    print("precision: {}".format(TP/(TP+FP)))
    return TP, samples_TP, FP, samples_FP, FN, samples_FN


def compare_GT_database_search(file, database_file="edited_ontology.json", include_sys=False):
    database = load_multiwoz_database(database_file)
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
                GT = [v for ds, v in turn['turn_label'] if ds not in ['hotel-parking', 'hotel-internet']]
                usr_res = find_database_value_in_utterance(turn['transcript'], database)
                results = [usr_res]
                if include_sys:
                    sys_res = find_database_value_in_utterance(turn['system_transcript'], database)
                    results.append(sys_res)

                VAL = set()
                for res in results:
                    for k, vs in res.items():
                        for v in vs:
                            VAL.add(v.strip())

                tmp_TP, tmp_samples_TP, tmp_FP, tmp_samples_FP, tmp_FN, tmp_samples_FN = compare_NER_IN_GT(GT, VAL)
                TP += tmp_TP
                samples_TP.extend(tmp_samples_TP)
                FP += tmp_FP
                samples_FP.extend(tmp_samples_FP)
                FN += tmp_FN
                samples_FN.extend(tmp_samples_FN)

                # if tmp_FN > 0:
                #     print(turn['transcript'])
                #     print(turn['turn_label'])
                #     print(VAL)
                #     print()

    samples_TP = gather_sort_samples(samples_TP)
    samples_FP = gather_sort_samples(samples_FP)
    samples_FN = gather_sort_samples(samples_FN)

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
