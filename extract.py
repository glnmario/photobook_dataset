import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from processor import Log
from collections import defaultdict, Counter
from transformers import AutoModel, AutoTokenizer


def load_logs(log_repository, data_path):

    filepath = os.path.join(data_path, log_repository)
    print("Loading logs from {}...".format(filepath))

    missing_counter = 0
    file_count = 0
    for _, _, files in os.walk(filepath):
        file_count += len(files)
    print("{} files found.".format(file_count))
    logs = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), 'r') as logfile:
                    log = Log(json.load(logfile))
                    if log.complete:
                        logs.append(log)

    print("DONE. Loaded {} completed game logs.".format(len(logs)))
    return logs


def collect_dataset(logs):
    labels = ["Game_ID", "Game_Domain_ID", "Game_Domain_1", "Game_Domain_2", "Game_Duration", "Game_Score", \
              "Feedback_A", "Feedback_B", 'Agent_1', "Agent_2", \
              "Round_Nr", "Round_Duration", "Round_Scores", "Round_Images_A", "Round_Images_B", \
              "Round_Common", "Round_Highlighted_A", "Round_Highlighted_B", \
              "Message_Nr", "Message_Timestamp", "Message_Turn", "Message_Agent_ID", \
              "Message_Speaker", "Message_Type", "Message_Text"]
    dataset = []
    for log in logs:
        game_data = [log.game_id, log.domain_id, log.domains[0], log.domains[1], log.duration.total_seconds(),
                     log.total_score, log.feedback["A"], log.feedback["B"], log.agent_ids[0], log.agent_ids[1]]
        for game_round in log.rounds:
            round_data = [game_round.round_nr - 1, game_round.duration.total_seconds(), game_round.total_score,
                          game_round.images["A"], game_round.images["B"], game_round.common,
                          game_round.highlighted["A"], game_round.highlighted["B"]]
            for message in game_round.messages:
                message_data = [message.message_id, message.timestamp, message.turn, message.agent_id, \
                                message.speaker, message.type, message.text]
                dataset.append(game_data + round_data + message_data)

    df = pd.DataFrame(dataset, columns=labels)

    return df


def remove_prefixing_zeros(s):
    if s[0] == 0:
        return remove_prefixing_zeros(s[1:])
    else:
        return s


def trim_image_url(s):
    s = s.split('/')[-1]  # remove folder path
    s = s.split('.')[0]  # remove extension
    return s


def cosine(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    return dot / (norma * normb)


def to_bow(hidden_states, normalize=True):
    bow = []
    for t in np.arange(hidden_states.shape[0]):
        h_t = np.copy(hidden_states[t, :])
        if normalize:
            h_t /= np.linalg.norm(h_t)
        bow.append(h_t)
    return bow


def mean_sentence_similarity(text, caption_reprs):
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        text_repr = last_hidden_states[0, 0, :].data.numpy()
        text_repr /= np.linalg.norm(text_repr)

    return np.mean([np.dot(text_repr, c) for c in caption_reprs])


def R_bert(reference, candidate, tf):
    R = 0
    if tf:
        tf_sum = 0

    for w_t, tf_t in reference:
        if tf:
            R += tf_t * np.max([np.dot(w_t, v_t) for v_t in candidate])
            tf_sum += tf_t
        else:
            R += np.max([np.dot(w_t, v_t) for v_t in candidate])

    if tf:
        R /= tf_sum

    return R


def P_bert(reference, candidate):
    P = 0
    for w_t in candidate:
        P += np.max([np.dot(w_t, v_t) for v_t in reference])
    return P


def F_bert(reference, candidate):
    P = P_bert(reference, candidate)
    R = R_bert(reference, candidate)
    return 2 * ((P * R) / (P + R))


def mean_bert_recall(text, caption_reprs, tf=True):
    """
    BERTScore: Zhang et al. 2019
    """
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        text_repr = last_hidden_states[0, 1:-1, :].data.numpy()
        candidate = to_bow(text_repr)

    Rs = [R_bert(ref, candidate, tf=tf) for ref in caption_reprs]
    return np.mean(Rs)



if __name__ == '__main__':

    LOAD_CHAINS = True
    LOAD_CAPTIONS = True
    LOAD_CAPTION_REPRESENTATIONS = True

    model = AutoModel.from_pretrained('bert-base-cased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    if LOAD_CHAINS:
        with open('new_chains.dict', 'rb') as f:
            all_chains = pickle.load(file=f)
    else:
        data_path = ""
        logs = load_logs("logs", data_path)
        dataset = collect_dataset(logs)

        F = ['Game_ID', 'Round_Nr', 'Round_Common', 'Message_Nr', 'Message_Speaker', 'Message_Type', 'Message_Text']
        f2i = {f: i for i, f in enumerate(F)}

        buffer = []
        current_game_id = -1
        recognised_as_common = []  # images
        prev_row_type = ''
        prev_img_id = ''
        tot_messages = 0
        all_chains = defaultdict(list)
        # unique_img_id_str = defaultdict(dict)

        for index, row in tqdm(dataset.iterrows(), total=len(dataset)):

            # collect fields of interest
            fields = {f: row[f] for f in F}

            # flush buffer when a new game starts
            if fields['Game_ID'] != current_game_id:
                buffer = []
                recognised_as_common = []
                current_game_id = fields['Game_ID']
                prev_row_type = ''
                prev_img_id = ''

            if fields['Message_Type'] == 'selection':
                # parse selection: <com>/<dif> + img_id_str
                _, selection, img_path = fields['Message_Text'].split(' ')
                # img_id_str = trim_image_url(img_path)
                img_id_str = img_path.split('/')[-1]
                # unique_img_id_str[img_id_str][img_path] = None

                if selection == '<com>' and img_path in fields['Round_Common'] and img_path not in recognised_as_common:
                    recognised_as_common.append(img_path)
                    all_chains[img_path].append(buffer)
                    prev_img_id = img_path

                elif img_path in recognised_as_common and img_path != prev_img_id:
                    all_chains[img_path][-1] += buffer
                    prev_img_id = img_path

            elif fields['Message_Type'] == 'text':
                tot_messages += 1
                if prev_row_type == 'selection':
                    buffer = []
                buffer.append((fields['Round_Nr'], fields['Message_Speaker'], fields['Message_Text']))
            else:
                pass

            prev_row_type = fields['Message_Type']

        # print('{} unique image ids found.'.format(len(unique_img_id_str)))
        # for id_string, img_paths in unique_img_id_str.items():
        #     if len(img_paths) > 1:
        #         print(id_string)
        #         print(img_paths)

        print('Save chains to disk.')
        with open('new_chains.dict', 'wb') as f:
            pickle.dump(all_chains, file=f)


    if LOAD_CAPTIONS:
        with open('annotations/photobook_captions.dict', 'rb') as f:
            all_captions = pickle.load(f)
    else:
        with open('annotations/captions_train2014.json', 'r') as f:
            annotations = json.loads(f.read())

        str2id, id2str = {}, {}
        img_strings = [path.split('/')[-1] for path in all_chains]
        for image_data in annotations['images']:
            # coco_id = trim_image_url(image_data['coco_url'])
            coco_id = image_data['coco_url'].split('/')[-1]
            # print(coco_id)
            # print(int(image_data['id']))
            if coco_id in img_strings:
                str2id[coco_id] = int(image_data['id'])
                id2str[int(image_data['id'])] = coco_id

        n_captions = 0
        all_captions = defaultdict(list)
        for ann in annotations['annotations']:
            try:
                all_captions[id2str[int(ann['image_id'])]].append(ann['caption'].strip())
                n_captions += 1
            except KeyError:
                continue

        print('{} captions collected for {} images.'.format(n_captions, len(all_captions)))
        with open('annotations/photobook_captions.dict', 'wb') as f:
            pickle.dump(all_captions, file=f)


    if LOAD_CAPTION_REPRESENTATIONS:
        with open('caption_representations_tf.dict', 'rb') as f:
            caption_reprs = pickle.load(f)
    else:
        caption_reprs = defaultdict(list)

        # for each image in our chains
        for img_path in tqdm(all_chains):

            img_id_str = img_path.split('/')[-1]

            # compute tf using the >= 5 reference captions as a document
            tf_counter = Counter()
            for caption in all_captions[img_id_str]:
                input_ids = torch.tensor([tokenizer.encode(caption, add_special_tokens=False)])
                tf_counter += Counter(input_ids[0].numpy())

            for caption in all_captions[img_id_str]:
                input_ids = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)])

                with torch.no_grad():
                    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
                    reprs = last_hidden_states[0, 1:-1, :].data.numpy()  # [0, 0, :] for CLS

                bow_reprs = to_bow(reprs)
                input_tfs = [tf_counter[w] for w in input_ids[0][1:-1].numpy()]
                caption_reprs[img_id_str].append(tuple(zip(bow_reprs, input_tfs)))

        print('Save encoded captions to disk.')
        with open('caption_representations_tf.dict', 'wb') as f:
            pickle.dump(caption_reprs, file=f)

    chains_with_sim = defaultdict(list)
    for img_path, cs in tqdm(all_chains.items()):
        img_id_str = img_path.split('/')[-1]
        new_cs = []
        for c in cs:
            new_c = []
            for round_nr, speaker, text in c:

                similarity = mean_bert_recall(text, caption_reprs[img_id_str])
                new_c.append((round_nr, speaker, text, similarity))
            new_cs.append(new_c)

        chains_with_sim[img_path] = new_cs

    with open('chains_mean_Rbert_tf.dict', 'wb') as f:
        pickle.dump(caption_reprs, file=f)
