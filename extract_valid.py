import os
import json
import pickle
import random
import torch
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from processor import Log
from copy import deepcopy
from collections import defaultdict, Counter
from transformers import AutoModel, AutoTokenizer
# from nltk.corpus import stopwords
from spacy.lang.en import stop_words
from nltk.translate.meteor_score import single_meteor_score as meteor


def collect_dataset(logs):
    labels = ["Game_ID", "Game_Domain_ID", "Game_Domain_1", "Game_Domain_2", \
              "Feedback_A", "Feedback_B", 'Agent_1', "Agent_2", \
              "Round_Nr", "Round_Images_A", "Round_Images_B", \
              "Round_Common", "Round_Highlighted_A", "Round_Highlighted_B", \
              "Message_Nr", "Message_Timestamp", "Message_Turn", "Message_Agent_ID", \
              "Message_Speaker", "Message_Type", "Message_Text", "Message_Referent"]
    dataset = []
    for game_id, log in logs.items():
        game_data = [game_id, log.domain_id, log.domains[0], log.domains[1],
                     log.feedback["A"], log.feedback["B"], log.agent_ids[0], log.agent_ids[1]]
        for game_round in log.rounds:
            round_data = [game_round.round_nr - 1,
                          game_round.images["A"], game_round.images["B"], game_round.common,
                          game_round.highlighted["A"], game_round.highlighted["B"]]
            for message in game_round.messages:
                message_data = [message.message_id, message.timestamp, message.turn, message.agent_id, \
                                message.speaker, message.type, message.text, message.referent]
                dataset.append(game_data + round_data + message_data)

    df = pd.DataFrame(dataset, columns=labels)

    return df


def text_to_bow(text, remove_stopwords, return_ids=False):
    if text == '':
        return None

    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        text_repr = last_hidden_states[0, 1:-1, :].data.numpy()
        bow = hidden_to_bow(text_repr)

    if return_ids:
        return input_ids, bow
    else:
        return bow


def hidden_to_bow(hidden_states, normalize=True):
    bow = []
    for t in np.arange(hidden_states.shape[0]):
        h_t = np.copy(hidden_states[t, :])
        if normalize:
            h_t /= np.linalg.norm(h_t)
        bow.append(h_t)
    return bow


def bert_recall(reference, candidate, tf):
    R = 0
    if tf:
        tf_sum = 0

    for tok_t, w_t, tf_t in reference:
        if tf:
            R += tf_t * np.max([np.dot(w_t, v_t) for v_t in candidate])
            tf_sum += tf_t
        else:
            R += np.max([np.dot(w_t, v_t) for v_t in candidate])

    if tf:
        R /= tf_sum
    else:
        R /= len(reference)

    return R


def mean_bert_recall(candidate, references, tf=True):
    """
    Inspired by BERTScore (Zhang et al. 2019)
    """
    Rs = [bert_recall(ref, candidate, tf=tf) for ref in references]
    return np.mean(Rs)


def stopwords_filter(text):
    filtered = []
    for tok in text.split(' '):
        if tok.lower() not in stopwords_en:
            filtered.append(tok)

    return ' '.join(filtered)


def chains_from_game_logs(logs, whole_round):
    dataset = collect_dataset(logs)

    F = ['Game_ID', 'Round_Nr', 'Message_Nr', 'Message_Speaker', 'Message_Type', 'Message_Text', 'Round_Common',
         'Round_Images_A', 'Round_Images_B', 'Message_Referent']

    chains = defaultdict(list)
    buffer = []
    recognised_as_common = set()
    tot_messages = 0
    current_round_id = -1
    current_game_id = -1

    end_indices = defaultdict(int)

    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):

        # Collect fields of interest
        fields = {field: row[field] for field in F}

        # New round!
        if (fields['Round_Nr'] != current_round_id) and buffer:

            # 1. store utterances from previous round
            for im in recognised_as_common:
                extended_buffer = []
                for tupl_idx, tupl in enumerate(buffer):
                    in_segment = whole_round or tupl_idx < end_indices[im]
                    extended_buffer.append(tupl + (in_segment,))

                chains[im] += extended_buffer

            # 2. reset and update
            buffer = []
            end_indices = defaultdict(int)
            current_round_id = fields['Round_Nr']

        if fields['Game_ID'] != current_game_id:
            recognised_as_common = set()
            current_game_id = fields['Game_ID']

        if fields['Message_Type'] == 'selection':
            # parse selection: <com>/<dif> + img_id_str
            _, selection, img_path = fields['Message_Text'].split(' ')

            # a common image was selected as common by one of the speakers
            if selection == '<com>' and img_path in fields['Round_Common']:
                recognised_as_common.add(img_path)
                if not whole_round:
                    end_indices[img_path] = len(buffer)

            if selection == '<dif>' and img_path in recognised_as_common:
                if not whole_round:
                    end_indices[img_path] = len(buffer)

        # Within round: store utterance
        if fields['Message_Type'] == 'text':
            tot_messages += 1
            visual_context = set(fields['Round_Images_A']) | set(fields['Round_Images_B'])
            buffer.append((fields['Game_ID'], fields['Round_Nr'], fields['Message_Nr'], fields['Message_Speaker'],
                           fields['Message_Text'], fields['Message_Referent'], visual_context))

    return chains


def preprocess_captions(image_paths, captions, remove_stopwords):
    caption_representations = defaultdict(list)
    tf_counter = dict()

    for img_path in tqdm(image_paths):
        img_id_str = img_path.split('/')[-1]

        # no need to preprocess the same captions twice
        if img_id_str in tf_counter:
            continue

        # -----------------------------------------------------------------------------------
        # Compute term frequency (tf) using an image's >=5 reference captions as document
        # -----------------------------------------------------------------------------------
        # 1. initialise counter with unknown lemma id '-1' (equivalent to add-1 smoothing)
        tf_counter[img_id_str] = Counter({-1: 1})

        # 2. collect word frequencies
        for caption in captions[img_id_str]:

            # # remove trailing period
            # if caption[-1] == '.':
            #     caption = caption[:-1]

            # tokenise caption
            caption = ' '.join(spacy_tokenizer(caption))

            if remove_stopwords:
                caption = stopwords_filter(caption)
            input_ids = torch.tensor([tokenizer.encode(caption, add_special_tokens=False)])
            tf_counter[img_id_str] += Counter(input_ids[0].numpy())

        # 3. normalise by total token count
        tot_tokens = sum(tf_counter[img_id_str].values())
        for w in tf_counter[img_id_str]:
            tf_counter[img_id_str][w] /= tot_tokens

        # -----------------------------------------------------------------------------------
        # Get BERT contextualised representations for all tokens of the >=5 image captions
        # -----------------------------------------------------------------------------------
        for caption in captions[img_id_str]:

            # # remove trailing period
            # if caption[-1] == '.':
            #     caption = caption[:-1]

            # tokenise caption
            caption = ' '.join(spacy_tokenizer(caption))

            if remove_stopwords:
                caption = stopwords_filter(caption)

            # 1. tokenise caption and turn tokens into token ids
            input_ids = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)])

            # 2. get last layer's hidden state for each token in the current caption
            with torch.no_grad():
                last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
                reprs = last_hidden_states[0, 1:-1, :].data.numpy()  # [0, 0, :] for CLS

            # 3. build a bag-of-contextualised-words representation of the current caption
            bow_reprs = hidden_to_bow(reprs, normalize=True)

            # 4. collect tf values for each token in the current caption
            input_tfs = [tf_counter[img_id_str][w] for w in input_ids[0][1:-1].numpy()]

            # 5. store bag-of-words representations together with tf values
            caption_representations[img_id_str].append(tuple(zip(caption, bow_reprs, input_tfs)))

    return caption_representations, tf_counter


def get_best_description(utterances):
    best_description = ('', [], None, 0)
    for utterance in utterances:
        if utterance[3] >= best_description[3]:
            best_description = utterance

    description = best_description[0]
    description_as_bow = best_description[2]

    description_tok_ids = best_description[1]
    input_tfs = [-1 for w in description_tok_ids]  # use dummy tf values

    return tuple(zip(description, description_as_bow, input_tfs))


def vg_score(text, img_path, vg_attributes, vg_relations, visual_context):
    target_attributes = vg_attributes[img_path]
    target_relations = vg_relations[img_path]

    all_features = target_attributes | target_relations

    visual_context -= {img_path}

    confounding_attributes = set()
    confounding_relations = set()
    for path in visual_context:
        confounding_attributes |= vg_attributes[path]
        confounding_relations |= vg_relations[path]

    discriminative_attributes = target_attributes - confounding_attributes
    discriminative_relations = target_relations - confounding_relations

    discriminative_features = discriminative_attributes | discriminative_relations
    meteor_score = meteor(' '.join(discriminative_features), text, gamma=0)

    return meteor_score, discriminative_features, all_features


def chains_with_utterance_scores(image_paths, chains, caption_reprs, remove_stopwords, descriptions_as_captions,
                                 vg_attributes=None, vg_relations=None):
    use_vg = (vg_attributes is not None) and (vg_relations is not None)

    _caption_reprs = deepcopy(caption_reprs)
    chains_with_scores = defaultdict(list)

    for img_path in tqdm(image_paths):

        # if img_path != 'chair_couch/COCO_train2014_000000399122.jpg':
        # continue

        # get image id as a string for compatibility
        img_id_str = img_path.split('/')[-1]

        if descriptions_as_captions:
            current_round = -1
            current_game = -1
            utterances_in_current_round = []

        new_c = []
        for game_id, round_nr, msg_nr, speaker, text, referent, vis_context, in_segment in chains[img_path]:

            if descriptions_as_captions and game_id != current_game:
                _caption_reprs = deepcopy(caption_reprs)
                current_game = game_id
                current_round = round_nr

            # tokenise text
            text = ' '.join(spacy_tokenizer(text))

            if remove_stopwords:
                text = stopwords_filter(text)

            try:
                # convert text to a bag-of-contextualised-words representations
                input_ids, utt_bow = text_to_bow(text, remove_stopwords, return_ids=True)
            except TypeError:
                # as a result of stopwords filtering, the text may be empty
                continue

            if in_segment:
                # compute recall using BERTScore (Zhang et al. 2019)
                score = mean_bert_recall(utt_bow, _caption_reprs[img_id_str], tf=False)

                discriminative_features, all_features = {}, {}
                if use_vg:
                    # compute METEOR using Visual Genome annotations
                    meteor_score, discriminative_features, all_features = vg_score(text, img_path,
                                                                                   vg_attributes,
                                                                                   vg_relations,
                                                                                   vis_context)
                    # score = score * (1 + meteor_score)
                    score = score + meteor_score
            else:
                score = -1

            if descriptions_as_captions:
                utterances_in_current_round.append((text, input_ids[0][1:-1].numpy(), utt_bow, score))

            # base case - first iteration
            if descriptions_as_captions and current_round == -1:
                current_round = round_nr

            # new round
            if descriptions_as_captions and current_round != round_nr:
                best_description_in_round = get_best_description(utterances_in_current_round)
                _caption_reprs[img_id_str].append(best_description_in_round)

                current_round = round_nr
                utterances_in_current_round = []

            # store current utterance with score
            new_c.append(
                (game_id, round_nr, msg_nr, speaker, text, score, discriminative_features, all_features, referent))

        # store current image's chains_3feb (with scores)
        chains_with_scores[img_path] += new_c

    return chains_with_scores


def get_captions(captions_path, chains):
    id2str = {}
    img_strings = [path.split('/')[-1] for path in chains]

    with open(captions_path, 'r') as f:
        annotations = json.loads(f.read())

    for image_data in annotations['images']:
        coco_id = image_data['coco_url'].split('/')[-1]
        if coco_id in img_strings:
            id2str[int(image_data['id'])] = coco_id

    n_captions = 0
    captions = defaultdict(list)
    for ann in annotations['annotations']:
        try:
            captions[id2str[int(ann['image_id'])]].append(ann['caption'].strip())
            n_captions += 1
        except KeyError:
            continue

    print('{} captions collected for {} images.'.format(n_captions, len(captions)))
    return captions


def main(output_path,
         whole_round,
         tf_weighting,
         remove_stopwords,
         descriptions_as_captions,
         use_vg,
         store_caption_representations=False,
         load_captions=True,
         limit=None,
         seed=0):
    random.seed(a=seed)

    if use_vg:
        with open('visual_genome/attributes.dict', 'rb') as f:
            vg_attributes = pickle.load(f)

        with open('visual_genome/relationships.dict', 'rb') as f:
            vg_relations = pickle.load(f)
    else:
        vg_attributes, vg_relations = None, None

    with open('logs/valid_logs.dict', 'rb') as f:
        valid_logs = pickle.load(f)

    valid_chains = chains_from_game_logs(valid_logs, whole_round=whole_round)

    if load_captions:
        with open('annotations/photobook_captions.dict', 'rb') as f:
            all_captions = pickle.load(f)
    else:
        all_captions = get_captions('annotations/captions_train2014.json', valid_chains)

        with open('annotations/photobook_captions.dict', 'wb') as f:
            pickle.dump(all_captions, file=f)

    # Collect all image paths, in case we only want to analyse some (`limit`) of them
    image_paths = list(valid_chains.keys())
    if limit:
        random.shuffle(image_paths)
        image_paths = image_paths[:limit]

    # Obtain bag-of-contextualised-words representations for all captions
    # as well as image-specific smoothed (add-1) term frequencies
    caption_reprs, tf_counter = preprocess_captions(image_paths, all_captions, remove_stopwords=True)

    if store_caption_representations:
        print('Save encoded captions to disk.')
        with open('caption_representations_tf.dict', 'wb') as f:
            pickle.dump(caption_reprs, file=f)

    if not tf_weighting:
        tf_counter = None

    # Score each utterance in the chains by its similarity to the respective reference captions
    chains_with_scores = chains_with_utterance_scores(
        image_paths, valid_chains, caption_reprs,
        remove_stopwords=remove_stopwords,
        descriptions_as_captions=descriptions_as_captions,
        vg_attributes=vg_attributes,
        vg_relations=vg_relations
    )

    # Store chains with scores
    with open('{}.dict'.format(output_path), 'wb') as f_out:
        pickle.dump(chains_with_scores, file=f_out)


if __name__ == '__main__':
    model = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    spacy_tokenizer = spacy.load("en_core_web_sm")

    stopwords_en = stop_words.STOP_WORDS
    stopwords_en |= {'sorry', 'noo', 'nope', 'oh', 'got'}
    stopwords_en -= {'above', 'across', 'against', 'all', 'almost', 'alone', 'along', 'among', 'amongst', 'at', 'back',
                     'behind', 'below', 'beside', 'between', 'beyond', 'bottom', 'down', 'eight', 'eleven', 'empty',
                     'few', 'fifteen', 'fifty', 'five', 'forty', 'four', 'front', 'full', 'hundred', 'he', 'him', 'his',
                     'himself', 'in', 'into', 'many', 'next', 'nine', 'nobody', 'none', 'noone', 'not', 'off', 'on',
                     'one', 'only', 'onto', 'out', 'over', 'part', 'several', 'side', 'she', 'her', 'herself', 'six',
                     'sixty', 'some', 'someone', 'something', 'somewhere', 'ten', 'their', 'them', 'themselves', 'they',
                     'three', 'through', 'thru', 'together', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two',
                     'under', 'up', 'used', 'using', 'various', 'very', 'with', 'within', 'without'}

    # stopwords_en = set(stopwords.words('english'))
    # stopwords_en |= {'sorry', 'yes', 'no', 'noo', 'nope', 'oh', 'got', 'this', 'that'}
    # stopwords_en -= {'above', 'against', 'below', 'between', 'down', 'further', 'in', 'into', 'off', 'on', 'out',
    #                  'over', 'through', 'under', 'up'}

    LIMIT = None
    SEED = 0

    # main('chains_valid_novg', whole_round=False, tf_weighting=False, remove_stopwords=False, use_vg=False,
    #     descriptions_as_captions=False, load_captions=True, limit=LIMIT, seed=SEED)

    # main('chains_valid_novg_nostopwords', whole_round=False, tf_weighting=False, remove_stopwords=True, use_vg=False,
    #      descriptions_as_captions=False, load_captions=True, limit=LIMIT, seed=SEED)
    #
    # main('chains_valid_vg', whole_round=False, tf_weighting=False, remove_stopwords=False, use_vg=True,
    #      descriptions_as_captions=False, load_captions=True, limit=LIMIT, seed=SEED)
    #
    # # main('chains_valid_vg_nostopwords', whole_round=False, tf_weighting=False, remove_stopwords=True, use_vg=True,
    #      descriptions_as_captions=False, load_captions=True, limit=LIMIT, seed=SEED)
    #
    # main('chains_valid_vg_nostopwords_wholeround', whole_round=True, tf_weighting=False, remove_stopwords=True, use_vg=True,
    #      descriptions_as_captions=False, load_captions=True, limit=LIMIT, seed=SEED)
    #
    # main('chains_test_vg_nostopwords_descr', whole_round=False, tf_weighting=False, remove_stopwords=True, use_vg=True,
    #      descriptions_as_captions=True, load_captions=True, limit=LIMIT, seed=SEED)

    main('chains_valid_vg_sum', whole_round=False, tf_weighting=False, remove_stopwords=False, use_vg=True,
         descriptions_as_captions=False, load_captions=True, limit=LIMIT, seed=SEED)
