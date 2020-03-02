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
from transformers import BertModel, BertTokenizer
# from nltk.corpus import stopwords
from spacy.lang.en import stop_words
from nltk.translate.meteor_score import single_meteor_score as meteor


def text_to_bow(text, stopwords=None):
    if text == '':
        return None

    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
        text_repr = last_hidden_states[0, 1:-1, :].data.numpy()
        bow = hidden_to_bow(text_repr)

    tokens = tokenizer.tokenize(text)
    assert len(tokens) == len(bow)

    if stopwords:
        bow_tmp = []
        tokens_tmp = []

        for tok, vec in zip(tokens, bow):
            if tok not in stopwords:
                bow_tmp.append(vec)
                tokens_tmp.append(tok)

        bow = bow_tmp
        tokens = tokens_tmp

    return tokens, input_ids, bow


def hidden_to_bow(hidden_states, normalize=True):
    bow = []
    for t in np.arange(hidden_states.shape[0]):
        h_t = np.copy(hidden_states[t, :])
        if normalize:
            h_t /= np.linalg.norm(h_t)
        bow.append(h_t)
    return bow


def bert_precision(reference, candidate, stopwords=None):
    if stopwords is None:
        stopwords = set()

    P = 0
    n_tokens = 0
    for w_t in candidate:
        n_tokens += 1
        cosines = []

        for tok_t, v_t in reference:
            if tok_t not in stopwords:
                cosines.append(np.dot(w_t, v_t))

        if cosines:
            P += np.max(cosines)

    # Normalise
    P = P / n_tokens if n_tokens else 0

    return P


def bert_recall(reference, candidate, stopwords=None):
    if stopwords is None:
        stopwords = set()

    R = 0
    n_tokens = 0
    for tok_t, w_t in reference:
        if tok_t in stopwords:
            continue
        else:
            n_tokens += 1

        cosines = [np.dot(w_t, v_t) for v_t in candidate]
        if cosines:
            R += np.max(cosines)

    # Normalise
    R = R / n_tokens if n_tokens else 0

    return R


def bert_f1(reference, candidate, stopwords=None):
    r = bert_recall(reference, candidate, stopwords)
    p = bert_precision(reference, candidate, stopwords)

    if p + r == 0:
        return 0
    else:
        return 2 * ((p * r) / (p + r))


def mean_bert_precision(references, candidate, stopwords=None):
    """
    Inspired by BERTScore (Zhang et al. 2019)
    """
    return np.mean([bert_precision(ref, candidate, stopwords=stopwords) for ref in references])


def mean_bert_recall(references, candidate, stopwords=None):
    """
    Inspired by BERTScore (Zhang et al. 2019)
    """
    return np.mean([bert_recall(ref, candidate, stopwords=stopwords) for ref in references])


def mean_bert_f1(references, candidate, stopwords=None):
    """
    Inspired by BERTScore (Zhang et al. 2019)
    """
    return np.mean([bert_f1(ref, candidate, stopwords=stopwords) for ref in references])


def stopwords_filter(text):
    filtered = []
    for tok in text.split(' '):
        if tok.lower() not in stopwords_en:
            filtered.append(tok)

    return ' '.join(filtered)


def preprocess_captions(image_paths, captions):
    caption_representations = defaultdict(list)

    for img_path in tqdm(image_paths):
        img_id_str = img_path.split('/')[-1]

        # -----------------------------------------------------------------------------------
        # Get BERT contextualised representations for all tokens of the >=5 image captions
        # -----------------------------------------------------------------------------------
        for caption in captions[img_id_str]:
            input_ids = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)])

            # 2. get last layer's hidden state for each token in the current caption
            with torch.no_grad():
                last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
                reprs = last_hidden_states[0, 1:-1, :].data.numpy()  # [0, 0, :] for CLS

            # 3. build a bag-of-contextualised-words representation of the current caption
            bow_reprs = hidden_to_bow(reprs, normalize=True)

            # 5. store bag-of-words representations
            caption_tokens = tokenizer.tokenize(caption)
            assert len(caption_tokens) == len(bow_reprs)
            caption_representations[img_id_str].append(tuple(zip(caption_tokens, bow_reprs)))

    return caption_representations


def get_best_description(utterances):
    best_description = ('', [], None, 0)
    for utterance in utterances:
        if utterance[3] >= best_description[3]:
            best_description = utterance

    description = best_description[0]
    description_as_bow = best_description[2]

    return tuple(zip(description, description_as_bow))


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


def chains_from_logs(logs):
    chains = defaultdict(list)

    for game_id, log in logs.items():

        tracked_in_game = set()

        for game_round in log.rounds:

            # 1. store utterances from previous round
            if game_round.round_nr > 1:
                for img in tracked_in_game:

                    tmp_buffer = []
                    any_utterance_in_segment = False
                    for _idx, utterance in enumerate(buffer):
                        tmp_utterance = deepcopy(utterance)
                        tmp_utterance['In_Segment'] = _idx < metadata[img]['last_index']
                        tmp_utterance['Reason'] = metadata[img]['reason']
                        tmp_buffer.append(tmp_utterance)
                        any_utterance_in_segment = any_utterance_in_segment or tmp_utterance['In_Segment']

                    if any_utterance_in_segment:
                        chains[img] += tmp_buffer

            buffer = []
            metadata = defaultdict(lambda: {'last_index': 0, 'reason': ''})

            for message in game_round.messages:

                if message.type == 'selection':
                    # parse selection: <com>/<dif> + img_id_str
                    _, selection, img_path = message.text.split(' ')

                    # a common image was selected as common by one of the speakers
                    if selection == '<com>' and img_path in game_round.common:
                        tracked_in_game.add(img_path)
                        metadata[img_path]['last_index'] = len(buffer)
                        metadata[img_path]['reason'] = '<com>'

                    if selection == '<dif>' and img_path in tracked_in_game:
                        metadata[img_path]['last_index'] = len(buffer)
                        metadata[img_path]['reason'] = '<dif>'

                elif message.type == 'text':
                    utterance = {'Game_ID': game_id, 'Round_Nr': game_round.round_nr, 'Message_Nr': message.message_id,
                                 'Message_Speaker': message.speaker, 'Message_Type': message.type,
                                 'Message_Text': message.text, 'Round_Common': game_round.common,
                                 'Round_Images_A': game_round.images['A'], 'Round_Images_B': game_round.images['B'],
                                 'Message_Referent': message.referent, 'Game_Domain_ID': log.domain_id,
                                 'Game_Domain_1': log.domains[0],
                                 'Game_Domain_2': log.domains[1], 'Feedback_A': log.feedback['A'],
                                 'Feedback_B': log.feedback['B'], 'Agent_1': log.agent_ids[0],
                                 'Agent_2': log.agent_ids[1], 'Round_Highlighted_A': game_round.highlighted['A'],
                                 'Round_Highlighted_B': game_round.highlighted['B'],
                                 'Message_Timestamp': message.timestamp, 'Message_Turn': message.turn,
                                 'Message_Agent_ID': message.agent_id}
                    buffer.append(utterance)

    return chains


def chains_with_utterance_scores(image_paths, chains, caption_reprs, remove_stopwords, bert_metric,
                                 remove_nondiscriminative_caption_words, descriptions_as_captions,
                                 vg_attributes=None, vg_relations=None):
    use_vg = (vg_attributes is not None) and (vg_relations is not None)

    _caption_reprs = deepcopy(caption_reprs)
    chains_with_scores = defaultdict(list)

    for img_path in tqdm(image_paths):

        # get image id as a string for compatibility
        img_id_str = img_path.split('/')[-1]

        if descriptions_as_captions:
            current_round = -1
            current_game = -1
            utterances_in_current_round = []

        new_c = []
        for fields in chains[img_path]:
            if descriptions_as_captions and fields['Game_ID'] != current_game:
                _caption_reprs = deepcopy(caption_reprs)
                current_game = fields['Game_ID']
                current_round = fields['Round_Nr']

            stopwords = set()
            if remove_stopwords:
                stopwords = stopwords_en

            caption_stopwords = set()
            if remove_nondiscriminative_caption_words:
                for full_img_path in set(fields['Round_Images_A']) | set(fields['Round_Images_B']) - {img_path}:
                    for caption_tuple in _caption_reprs[full_img_path.split('/')[-1]]:
                        caption_tokens = [c_tok for c_tok, _, in caption_tuple]
                        caption_stopwords |= set(caption_tokens)

            try:
                # convert text to a bag-of-contextualised-words representations
                tokens, input_ids, utt_bow = text_to_bow(fields['Message_Text'], stopwords)
                fields['Tokens'] = tokens
            except TypeError:
                # as a result of stopwords filtering, the text may be empty
                continue

            fields['Score'] = -1
            discriminative_features, all_features = {}, {}
            if fields['In_Segment']:
                # compute recall using BERTScore (Zhang et al. 2019)
                if bert_metric == 'recall':
                    bert_score_fn = mean_bert_recall
                elif bert_metric == 'precision':
                    bert_score_fn = mean_bert_precision
                elif bert_metric == 'f1':
                    bert_score_fn = mean_bert_f1
                else:
                    raise ValueError('Invalid BERT metric:', bert_metric)

                bert_score = bert_score_fn(_caption_reprs[img_id_str], utt_bow,
                                           stopwords=(stopwords | caption_stopwords))

                fields['Bert_Score'] = bert_score
                fields['Score'] = bert_score
                if use_vg:
                    visual_context = set(fields['Round_Images_A']) | set(fields['Round_Images_B'])

                    # preprocess text
                    text_for_vg = ' '.join([tok.text for tok in spacy_tokenizer(fields['Message_Text'])])
                    if remove_stopwords:
                        text_for_vg = stopwords_filter(text_for_vg)

                    # compute METEOR using Visual Genome annotations
                    meteor_score, discriminative_features, all_features = vg_score(text_for_vg, img_path,
                                                                                   vg_attributes,
                                                                                   vg_relations,
                                                                                   visual_context)
                    # score = bert_score * (1 + meteor_score)
                    fields['Meteor_Score'] = meteor_score
                    fields['Score'] += meteor_score

            fields['Discriminative_Features'] = discriminative_features
            fields['All_features'] = all_features

            if descriptions_as_captions:
                utterances_in_current_round.append((tokens, input_ids[0][1:-1].numpy(), utt_bow, fields['Score']))

            # base case - first iteration
            if descriptions_as_captions and current_round == -1:
                current_round = fields['Round_Nr']

            # new round
            if descriptions_as_captions and current_round != fields['Round_Nr']:
                best_description_in_round = get_best_description(utterances_in_current_round)
                _caption_reprs[img_id_str].append(best_description_in_round)

                current_round = fields['Round_Nr']
                utterances_in_current_round = []

            # store current utterance with score
            new_c.append(fields)

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
         remove_stopwords,
         remove_caption_words,
         bert_metric,
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

    with open('logs/dev_logs.dict', 'rb') as f:
        dev_logs = pickle.load(f)

    dev_chains = chains_from_logs(dev_logs)

    if load_captions:
        with open('annotations/photobook_captions.dict', 'rb') as f:
            all_captions = pickle.load(f)
    else:
        all_captions = get_captions('annotations/captions_train2014.json', dev_chains)

        with open('annotations/photobook_captions.dict', 'wb') as f:
            pickle.dump(all_captions, file=f)

    # Collect all image paths, in case we only want to analyse some (`limit`) of them
    image_paths = list(dev_chains.keys())
    if limit:
        random.shuffle(image_paths)
        image_paths = image_paths[:limit]

    # Obtain bag-of-contextualised-words representations for all captions
    # as well as image-specific smoothed (add-1) term frequencies
    caption_representations = preprocess_captions(image_paths, all_captions)

    if store_caption_representations:
        print('Save encoded captions to disk.')
        with open('caption_representations.dict', 'wb') as f:
            pickle.dump(caption_representations, file=f)

    # Score each utterance in the chains by its similarity to the respective reference captions
    chains_with_scores = chains_with_utterance_scores(
        image_paths, dev_chains, caption_representations,
        bert_metric=bert_metric,
        remove_stopwords=remove_stopwords,
        remove_nondiscriminative_caption_words=remove_caption_words,
        descriptions_as_captions=descriptions_as_captions,
        vg_attributes=vg_attributes,
        vg_relations=vg_relations
    )

    # Store chains with scores
    with open('{}.dict'.format(output_path), 'wb') as f_out:
        pickle.dump(chains_with_scores, file=f_out)


if __name__ == '__main__':
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # # Because of new error with transformers 2.5.0
    # tokenizer.pad_token = '<PAD>'
    # model.resize_token_embeddings(len(tokenizer))

    spacy_tokenizer = spacy.load('en_core_web_sm')

    stopwords_en = stop_words.STOP_WORDS
    stopwords_en |= {'sorry', 'noo', 'nope', 'oh', 'got', 'ha', '.', '!', '?', ','}
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

    FIELDS = ['Game_ID', 'Round_Nr', 'Message_Nr', 'Message_Speaker', 'Message_Type', 'Message_Text', 'Round_Common',
              'Round_Images_A', 'Round_Images_B', 'Message_Referent', 'Game_Domain_ID', 'Game_Domain_1',
              'Game_Domain_2', 'Feedback_A', 'Feedback_B', 'Agent_1', 'Agent_2', 'Round_Highlighted_A',
              'Round_Highlighted_B', 'Message_Timestamp', 'Message_Turn', 'Message_Agent_ID']

    LIMIT = None
    SEED = 0

    #
    # main('chains/dev_pr_keep1', remove_stopwords=False, use_vg=False, remove_caption_words=False,
    #      descriptions_as_captions=True, load_captions=False, limit=LIMIT, seed=SEED, bert_metric='precision')

    main('chains/dev_pr+vg_keep1', remove_stopwords=False, use_vg=True, remove_caption_words=False,
         descriptions_as_captions=True, load_captions=False, limit=LIMIT, seed=SEED, bert_metric='precision')

    main('chains/dev_f1_keep1', remove_stopwords=False, use_vg=False, remove_caption_words=False,
         descriptions_as_captions=True, load_captions=False, limit=LIMIT, seed=SEED, bert_metric='f1')

    main('chains/dev_f1+vg_keep1', remove_stopwords=False, use_vg=True, remove_caption_words=False,
         descriptions_as_captions=True, load_captions=False, limit=LIMIT, seed=SEED, bert_metric='f1')

    main('chains/dev_re_keep1', remove_stopwords=False, use_vg=False, remove_caption_words=False,
         descriptions_as_captions=True, load_captions=False, limit=LIMIT, seed=SEED, bert_metric='recall')

    main('chains/dev_re+vg_keep1', remove_stopwords=False, use_vg=True, remove_caption_words=False,
         descriptions_as_captions=True, load_captions=False, limit=LIMIT, seed=SEED, bert_metric='recall')
