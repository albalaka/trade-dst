import utils.multiwoz_dataset as multiwoz_dataset
import os
import json
import random
import pickle as pkl
from torch.utils.data import DataLoader
from torch import cuda
from embeddings import GloveEmbedding, KazumaCharEmbedding
from BertForValueExtraction import BertForValueExtraction
from transformers import BertTokenizer
from tqdm import tqdm


import en_core_web_sm
ner = en_core_web_sm.load()

# Main differences:
#   default tokens in different order, additional ENT_token

# Places to add ENTITIES:
#   Dataset? Lang? collate fn?

# Improvements?
#   In read_langs() instead of separating by ";", separate by USR_token, SYS_token?
#   Remove mem_lang???
#   in alon_TRADE.TRADE.encode_and_decode() do we need the random mask at all??
#   same as above, in Generator.forward() can we get rid of words_class*
#   in alon_TRADE.TRADE.compute_slot_acc() they may have been calculating accuracy incorrectly??

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


class Lang():
    """
    Class to hold a vocabulary, along with a mapping from
        english -> token index
        token index -> english
    """

    def __init__(self, PAD_token, SOS_token, EOS_token, UNK_token, ENT_token):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS",
                           EOS_token: "EOS", UNK_token: "UNK",
                           ENT_token: "ENT"}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sent, type):
        """Add words to language"""
        if type == 'utter':
            for word in sent.split(" "):
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
                for v in value.split(" "):
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def read_language(dataset_path, gating_dict, slots, dataset, language, mem_language,
                  ENT_token=None, appended_labels=None, percent_ground_truth=100, only_domain='',
                  except_domain='', data_ratio=100, drop_slots=None):
    """ Load a dataset of dialogues and add utterances, slots, domains
    :param dataset_path: path to a json dataset (rg. data/train_dials.json)
    :param gating_dict: dict with mapping for gating mechanism (ptr, dont care, none)
    :param slots: all domain-slots
    :param dataset: train, dev, or test
    :param language: Lang class for utterances
    :param mem_language: Lang class, for belief states
    :param only_domain: specify if training/testing on a single domain
    :param except_domain: specify if training/testing on all except a specific domain
    """

    print("READING DATASET")
    data = []
    max_response_len, max_value_len = 0, 0
    domain_counter = {}

    # Load all dialogues in the dataset
    dialogues = json.load(open(dataset_path))

    # If we need the BERT_VE model, load it, and the tokenizer
    if appended_labels == 'BERT_VE':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        ve_model = BertForValueExtraction(from_pretrained='BERT_ValueExtraction_models/bs30_gradacc20_lr1e-6_fp16_freezebert-ACC0.8590')
        if cuda.is_available():
            ve_model.to('cuda')
            ve_model.eval()

    # create the vocab for this dataset
    for dialogue_dict in dialogues:
        for turn in dialogue_dict['dialogue']:
            language.index_words(turn['system_transcript'], 'utter')
            language.index_words(turn['transcript'], 'utter')

    # For only using a portion of total data
    if data_ratio != 100:
        random.Random(10).shuffle(dialogues)
        dialogues = dialogues[:int(len(dialogues)*data_ratio*0.01)]

    for dialogue_dict in tqdm(dialogues):
        dialogue_history = ""

        # Filter domains - maybe?
        for domain in dialogue_dict['domains']:
            if domain not in EXPERIMENT_DOMAINS:
                continue
            if domain not in domain_counter.keys():
                domain_counter[domain] = 0
            domain_counter[domain] += 1

        # For training/testing on separate domains
        if only_domain and only_domain not in dialogue_dict['domains']:
            continue
        if (except_domain and dataset == 'test' and except_domain not in dialogue_dict['domains']) \
                or (except_domain and dataset != 'test' and except_domain in dialogue_dict['domains']):
            continue

        # Read dialogue data
        for turn in dialogue_dict['dialogue']:
            turn_domain = turn['domain']
            turn_idx = turn['turn_idx']
            # turn_utterance = turn['system_transcript']+" ; "+turn['transcript']
            # turn_utterance_stripped = turn_utterance.strip()
            dialogue_history += (turn['system_transcript'] +
                                 " ; "+turn['transcript'])

            # ADD grount truth labels from this turn
            if appended_labels == 'ground_truth':
                for domain_slot, value in turn['turn_label']:
                    if random.random() <= percent_ground_truth*0.01:
                        dialogue_history += f" {ENT_token} {value}"

            # IF using NER, add NER labels here
            if appended_labels == "NER":
                # append entities from the user utterance
                res = ner(turn['transcript'])
                for word in res:
                    if word.ent_iob_ == "B":
                        dialogue_history += f" {ENT_token} {word}"
                    if word.ent_iob == "I":
                        dialogue_history += f" {word}"

            # Use NER labels + GT labels for binary slots
            if appended_labels == "boosted_NER":
                res = ner(turn['transcript'])
                for word in res:
                    if word.ent_iob_ == "B":
                        dialogue_history += f" {ENT_token} {word}"
                    if word.ent_iob == "I":
                        dialogue_history += f" {word}"
                for domain_slot, value in turn['turn_label']:
                    if domain_slot in ['hotel-parking', 'hotel-internet']:
                        dialogue_history += f" {ENT_token} {value}"

            if appended_labels == "BERT_VE":
                values = ve_model.predict_sentence_values(tokenizer, turn['transcript'])
                for value in values:
                    dialogue_history += f" {ENT_token} {value}"

            dialogue_history += " ; "
            source_text = dialogue_history.strip()
            turn_belief_dict = fix_general_label_error(turn['belief_state'], slots, drop_slots)

            # For training/testing on separate domains
            # Generate domain-dependent slot list
            slot_temp = slots
            if dataset == "train" or dataset == "dev":
                if except_domain != "":
                    slot_temp = [
                        k for k in slots if except_domain not in k]
                    turn_belief_dict = dict(
                        [(k, v) for k, v in turn_belief_dict.items() if except_domain not in k])
                elif only_domain != "":
                    slot_temp = [
                        k for k in slots if only_domain in k]
                    turn_belief_dict = dict(
                        [(k, v) for k, v in turn_belief_dict.items() if only_domain in k])
            else:
                if except_domain != "":
                    slot_temp = [
                        k for k in slots if except_domain in k]
                    turn_belief_dict = dict(
                        [(k, v) for k, v in turn_belief_dict.items() if except_domain in k])
                elif only_domain != "":
                    slot_temp = [
                        k for k in slots if only_domain in k]
                    turn_belief_dict = dict(
                        [(k, v) for k, v in turn_belief_dict.items() if only_domain in k])

            if drop_slots:
                turn_belief_list = []
                for k, v in turn_belief_dict.items():
                    if k not in drop_slots:
                        turn_belief_list.append(f"{k}-{v}")
            else:
                turn_belief_list = [f"{k}-{v}" for k, v in turn_belief_dict.items()]

            # if dataset == 'train':
            mem_language.index_words(turn_belief_dict, 'belief')

            generate_y, gating_label = [], []
            for slot in slot_temp:
                if slot in turn_belief_dict.keys():
                    slot_value = turn_belief_dict[slot]
                    generate_y.append(slot_value)

                    if slot_value == "dontcare":
                        gating_label.append(gating_dict[slot_value])
                    elif slot_value == "none":
                        gating_label.append(gating_dict[slot_value])
                    else:
                        gating_label.append(gating_dict['ptr'])

                    if max_value_len < len(turn_belief_dict[slot]):
                        max_value_len = len(turn_belief_dict[slot])

                else:
                    generate_y.append("none")
                    gating_label.append(gating_dict['none'])

            data_detail = {
                "ID": dialogue_dict["dialogue_idx"],
                # "domains": dialogue_dict["domains"], # never used
                "turn_domain": turn_domain,
                "turn_id": turn_idx,
                "dialog_history": source_text,
                "turn_belief": turn_belief_list,
                "gating_label": gating_label,
                # "turn_uttr": turn_utterance_stripped, # never used
                'generate_y': generate_y
            }
            data.append(data_detail)

            if max_response_len < len(source_text.split()):
                max_response_len = len(source_text.split())

    if "t{}".format(max_value_len-1) not in mem_language.word2index.keys() and dataset == "train":
        for time_i in range(max_value_len):
            mem_language.index_words("t{}".format(time_i), "utter")

    print("domain counter", domain_counter)
    return data, max_response_len, slot_temp


def get_sequence_dataloader(data, language, mem_language, batch_size, shuffle=True,
                            num_workers=0, pin_memory=False):
    data_keys = data[0].keys()
    data_info = {k: [] for k in data_keys}

    for datum in data:
        for k in data_keys:
            data_info[k].append(datum[k])

    dataset = multiwoz_dataset.Dataset(
        data_info, language.word2index, language.word2index, mem_language.word2index)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size, shuffle=shuffle,
                             collate_fn=multiwoz_dataset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return data_loader


def dump_pretrained_emb(word2index, index2word, dump_path):
    print(f"Dumping pretrained embeddings at {dump_path}")
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in range(len(word2index.keys())):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)


def get_slot_information(ontology, drop_slots=[]):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    slots = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    slots = [slot for slot in slots if slot not in drop_slots]
    return slots


def prepare_data(training, **kwargs):
    file_train = 'data/train_dials.json'
    file_dev = 'data/dev_dials.json'
    file_test = 'data/test_dials.json'

    batch_size = kwargs['MAX_GPU_SAMPLES']
    load_embeddings = kwargs['load_embedding']
    lang_path = kwargs['lang_path']

    if not os.path.exists(lang_path):
        os.makedirs(lang_path)

    # load domain-slot pairs from ontology
    ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    all_slots = get_slot_information(ontology, kwargs['drop_slots'])
    # all_slots = get_slot_information(ontology)
    gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}

    # Vocabulary
    lang = Lang(
        PAD_token=kwargs['PAD_token'],
        SOS_token=kwargs['SOS_token'],
        EOS_token=kwargs['EOS_token'],
        UNK_token=kwargs['UNK_token'],
        ENT_token=kwargs['ENT_token']
    )
    mem_lang = Lang(
        PAD_token=kwargs['PAD_token'],
        SOS_token=kwargs['SOS_token'],
        EOS_token=kwargs['EOS_token'],
        UNK_token=kwargs['UNK_token'],
        ENT_token=kwargs['ENT_token']
    )
    lang.index_words(all_slots, 'slot')
    mem_lang.index_words(all_slots, 'slot')
    lang_name = 'lang-all.pkl'
    mem_lang_name = 'mem-lang-all.pkl'

    if training:
        # Get training data, longest training turn length, slots used in training
        data_train, max_len_train, slot_train = read_language(file_train, gating_dict, all_slots, "train", lang, mem_lang,
                                                              ENT_token=lang.index2word[kwargs['ENT_token']],
                                                              appended_labels=kwargs['appended_labels'],
                                                              percent_ground_truth=kwargs['percent_ground_truth'],
                                                              data_ratio=kwargs['train_data_ratio'],
                                                              drop_slots=kwargs['drop_slots'])
        dataloader_train = get_sequence_dataloader(data_train, lang, mem_lang, batch_size)
        vocab_size_train = lang.n_words

        # Get dev data, longest dev turn length, slots used in dev
        data_dev, max_len_dev, slot_dev = read_language(file_dev, gating_dict, all_slots, "dev", lang, mem_lang,
                                                        ENT_token=lang.index2word[kwargs['ENT_token']],
                                                        appended_labels=kwargs['appended_labels'],
                                                        percent_ground_truth=kwargs['percent_ground_truth'],
                                                        data_ratio=kwargs['dev_data_ratio'],
                                                        drop_slots=kwargs['drop_slots'])
        dataloader_dev = get_sequence_dataloader(data_dev, lang, mem_lang, batch_size)

        data_test, max_len_test, slot_test = read_language(file_test, gating_dict, all_slots, "test", lang,
                                                           mem_lang, data_ratio=kwargs['test_data_ratio'],
                                                           drop_slots=kwargs['drop_slots'])
        dataloader_test = []

        # if language files already exist, load them
        if os.path.exists(os.path.join(lang_path, lang_name)) and os.path.exists(os.path.join(lang_path, mem_lang_name)):
            print(f"Loading saved language files from {os.path.join(lang_path, lang_name)}")
            with open(os.path.join(lang_path, lang_name), 'rb') as p:
                lang = pkl.load(p)
            with open(os.path.join(lang_path, mem_lang_name), 'rb') as p:
                mem_lang = pkl.load(p)

        # else dump the newly calculated languages
        else:
            print(f"Dumping language files to {os.path.join(lang_path, lang_name)}")
            with open(os.path.join(lang_path, lang_name), 'wb') as p:
                pkl.dump(lang, p)
            with open(os.path.join(lang_path, mem_lang_name), 'wb') as p:
                pkl.dump(mem_lang, p)

        # dump the pre-calculated embeddings for the language
        embedding_dump_path = f'data/emb{lang.n_words}.json'
        if not os.path.exists(embedding_dump_path) and load_embeddings:
            dump_pretrained_emb(lang.word2index, lang.index2word, embedding_dump_path)

    # if testing
    else:
        with open(os.path.join(lang_path, lang_name), 'rb') as handle:
            lang = pkl.load(handle)
        with open(os.path.join(lang_path, mem_lang_name), 'rb') as handle:
            mem_lang = pkl.load(handle)

        # set training and dev info to and 0's and empty
        data_train, max_len_train, slot_train, dataloader_train, vocab_size_train = [], 0, [], [], 0

        data_dev, max_len_dev, slot_dev, dataloader_dev = [], 0, [], []

        # Get test data, longest test turn length, slots used in test
        data_test, max_len_test, slot_test = read_language(file_test, gating_dict, all_slots, "test",
                                                           lang, mem_lang,
                                                           ENT_token=lang.index2word[kwargs['ENT_token']],
                                                           appended_labels=kwargs['appended_labels'],
                                                           percent_ground_truth=kwargs['percent_ground_truth'],
                                                           data_ratio=kwargs['test_data_ratio'],
                                                           drop_slots=kwargs['drop_slots'])

        dataloader_test = get_sequence_dataloader(data_test, lang, mem_lang, batch_size)

    max_word = max(max_len_train, max_len_dev, max_len_test) + 1

    print("Read %s pairs train" % len(data_train))
    print("Read %s pairs dev" % len(data_dev))
    print("Read %s pairs test" % len(data_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % vocab_size_train)
    print("Vocab_size Belief %s" % mem_lang.n_words)
    print("Max. length of dialog words for RNN: %s " % max_word)

    slots_list = [all_slots, slot_train, slot_dev, slot_test]

    print(
        f"[Train Set & Dev Set Slots]: Number is {len(slots_list[2])} in total")
    print(f"[Train Set & Dev Set Slots]: {slots_list[2]}")

    print(f"[Test Set Slots]: Number is {len(slots_list[3])} in total")
    print(f"[Test Set Slots]: {slots_list[3]}")

    langs = [lang, mem_lang]
    return dataloader_train, dataloader_dev, dataloader_test, langs, slots_list, gating_dict, vocab_size_train


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
