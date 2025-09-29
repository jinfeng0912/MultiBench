"""Implements BERT embedding extractors (Adapted for Windows MOSI)."""
import torch
from torch import nn
from transformers import AutoTokenizer, pipeline, BertModel
import h5py
import pickle
import numpy as np
import sys
import os  # Added for cross-platform path handling

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# features_extractor = pipeline('feature-extraction', model=model_name, tokenizer=model_name)
bert = BertModel.from_pretrained(model_name)
bert.config.output_hidden_states = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")
bert.to(device)

def get_bert_features(all_text, contextual_embedding=False, batch_size=64, max_len=None):
    """Get bert features from data.

    Use pipline to extract all the features, (num_points, max_seq_length, feature_dim): np.ndarray

    Args:
        all_text (list): Data to get BERT features from
        contextual_embedding (bool, optional): If True output the last hidden state of bert. If False, output the embedding of words. Defaults to False.
        batch_size (int, optional): Batch size. Defaults to 500.
        max_len (int, optional): Maximum length of the dataset. Defaults to None.

    Returns:
        np.array: BERT features of text.
    """
    output_bert_features = []
    if max_len == None:
        max_len = max([len([ms for ms in s.split() if len(ms) > 0]) for s in all_text])
    print(max_len)
    print(len(all_text))

    for i in range(0, len(all_text), batch_size):

        inputs = tokenizer(all_text[i: i+batch_size], padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")

        inputs = {key: value.to(device) for key, value in inputs.items()}

        bert_feartures = bert(**inputs)

        outputs = bert_feartures.hidden_states
        if contextual_embedding:
            output_bert_features.append(outputs[-1].detach().cpu().numpy())
        else:
            output_bert_features.append(outputs[0].detach().cpu().numpy())
            print(outputs[0].detach().cpu().numpy().shape)
        print('i = {} finished!'.format(i))

    print(np.concatenate(output_bert_features).shape)
    return np.concatenate(output_bert_features)


def get_rawtext(path, data_kind, vids=None):
    """"Get raw text from the datasets.

    Args:
        path (str): Path to data
        data_kind (str): Data Kind. Must be 'hdf5'.
        vids (list, optional): List of video data as np.array. Defaults to None.

    Returns:
        tuple(list, list): Text data list, video data list
    """
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
    else:
        with open(path, 'rb') as f_r:  # Ensured 'rb' for Windows
            f = pickle.load(f_r)
    text_data = []
    new_vids = []

    if vids == None:
        vids = list(f.keys())

    for vid in vids:
        text = []
        # If data IDs are NOT the same as the raw ids
        # add some code to match them here, eg. from vanvan_10 to vanvan[10]
        # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
        # vid_id = '{}[{}]'.format(id, seg)
        vid_id = int(vid[0]) if type(vid) == np.ndarray else vid
        try:
            if data_kind == 'hdf5':
                for word in f['words'][vid_id]['features']:
                    if word[0] != b'sp':
                        text.append(word[0].decode('utf-8'))
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
            else:
                for word in f[vid_id]:
                    if word != 'sp':
                        text.append(word)
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
        except:
            print("missing", vid, vid_id)
    return text_data, new_vids



def max_seq_len(id_list, max_len=50):
    """ Fix dataset to max sequence length.

    Cut the id lists with the max length, but didnt do padding here.
    Add the first one as [CLS] and the last one for [SEP].

    Args:
        id_list (list): List of ids to manipulate
        max_len (int, optional): Maximum sequence length. Defaults to 50.

    Returns:
        list: List of tokens
    """
    new_id_list = []
    for id in id_list:
        if len(id) > 0:
            id.append(id[-1])  # [SEP]
            id.insert(0, id[0]) # [CLS]
        new_id_list.append(id[:max_len])
    return new_id_list



def corresponding_other_modality_ids(orig_text, tokenized_text):
    """Align word ids to other modalities.

    Since tokenizer splits the word into parts e.g. '##ing' or 'you're' -> 'you', ''', 're'
    we should get the corresponding ids for other modalities' features applied to modalities
    which aligned to words

    Args:
        orig_text (list):  List of strings corresponding to the original text.
        tokenized_text (list): List of lists of tokens.

    Returns:
        list: List of ids.
    """
    id_list = []
    idx = -1
    for i, t in enumerate(tokenized_text):
        if '##' in t:  # deal with BERT sub words
            id_list.append(idx)
        elif '\'' == t:
            id_list.append(idx)
            if i+1 < len(tokenized_text):  # deal with [she's] [you're] [you'll] etc. or [sisters' parents] [brothers']
                if ''.join([tokenized_text[i-1], t, tokenized_text[i+1]]) in orig_text or tokenized_text[i+1] == 's':
                    idx -= 1
        elif '-' == t:  # deal with e.g. [good-time]
            id_list.append(idx)
            idx -= 1
        elif '{' == t:  # deal with {lg} and {cg} marks
            id_list.append(idx+1)
        elif '}' == t:
            id_list.append(idx)
        else:
            idx += 1
            id_list.append(idx)
    if len(id_list) > 0:
        ori_list = [k.strip() for k in orig_text.split(' ') if len(k) > 0]
        if len(ori_list) != id_list[-1]+1:
            print(orig_text)
            print(tokenized_text)
            print(id_list)
    return id_list


def bert_version_data(data, raw_path, keys, max_padding=50, bert_max_len=None):
    """Get bert encoded data

    Args:
        data (dict): Data dictionary
        raw_path (str): Path to raw data
        keys (dict): List of keys in raw text getter
        max_padding (int, optional): Maximum padding to add to list. Defaults to 50.
        bert_max_len (int, optional): Maximum length in BERT. Defaults to None.

    Returns:
        dict: Dictionary from modality to data.
    """

    file_type = raw_path.split('.')[-1]
    sarcasm_text, _ = get_rawtext(raw_path, file_type, keys)

    bert_features = get_bert_features(sarcasm_text, contextual_embedding=True, max_len=bert_max_len)  # (N, MAX_LEN, 768) for sarcasm

    # get corresponding ids
    other_modality_ids = []
    for origi_text in sarcasm_text:
        tokenized_sequence = tokenizer.tokenize(origi_text)
        other_modality_ids.append(corresponding_other_modality_ids(origi_text, tokenized_sequence))

    # apply max seq len, DON'T FORGET [CLS] and [SEP] token
    new_other_mids = max_seq_len(other_modality_ids, max_len=max_padding)

    # get other modal features and pad them to max len
    new_vision = []
    for i, v in enumerate(data['vision']):
        tmp = v[new_other_mids[i]]
        tmp = np.pad(tmp, ((0, max_padding - tmp.shape[0]), (0, 0)))
        new_vision.append(tmp)
    new_vision = np.stack(new_vision)

    new_audio = []
    for i, a in enumerate(data['audio']):
        tmp = a[new_other_mids[i]]
        tmp = np.pad(tmp, ((0, max_padding - tmp.shape[0]), (0, 0)))
        new_audio.append(tmp)
    new_audio = np.stack(new_audio)

    new_bert_features = []
    if bert_features.shape[1] >= max_padding:
        for b in bert_features:
            new_bert_features.append(b[:max_padding, :])
    else:
        for b in bert_features:
            new_bert_features.append(np.pad(b, ((0, max_padding-bert_features.shape[1]), (0, 0))))
    new_bert_features = np.stack(new_bert_features)

    return {'vision': new_vision, 'audio': new_audio, 'text': new_bert_features}


if __name__ == '__main__':
    # Windows MOSI paths (adapted; ensure files exist)
    input_pkl_path = r'E:\Laboratory\datasets\CMU_MOSI\mosi_raw.pkl'  # Input raw pkl
    raw_text_hdf5_path = r'E:\Laboratory\datasets\CMU_MOSI\mosi.hdf5'  # Raw text HDF5
    output_pkl_path = r'E:\Laboratory\datasets\CMU_MOSI\mosi_bert.pkl'  # Output (overwrites if exists)

    # Load input pkl
    try:
        with open(input_pkl_path, 'rb') as f:  # 'rb' for Windows binary
            alldata = pickle.load(f)
        print(f"Loaded input: {input_pkl_path}")
    except FileNotFoundError:
        print(f"Error: {input_pkl_path} not found. Run preprocess raw first.")
        sys.exit(1)
    except Exception as e:
        print(f"Load error: {e}")
        sys.exit(1)

    # Process each fold (train, valid, test) - adapted from original sarcasm example
    for fold in alldata.keys():
        print(f"\n--- Processing '{fold}' data ---")
        keys = list(alldata[fold]['id'])  # List of IDs (handle bytes if needed)
        print(f"{fold} vision shape: {alldata[fold]['vision'].shape}")  # Debug original

        # Process fold with bert_version_data (HDF5 for raw text)
        new_fold_data = bert_version_data(
            data=alldata[fold],
            raw_path=raw_text_hdf5_path,  # Use HDF5 for MOSI raw text
            keys=keys,
            max_padding=50,  # Keep original
            bert_max_len=None  # Dynamic from data
        )

        if new_fold_data is not None:
            alldata[fold]['vision'] = new_fold_data['vision']
            alldata[fold]['audio'] = new_fold_data['audio']
            alldata[fold]['text'] = new_fold_data['text']
            print(f"New {fold} shapes - vision: {new_fold_data['vision'].shape}, audio: {new_fold_data['audio'].shape}, text: {new_fold_data['text'].shape}")
        else:
            print(f"Error processing '{fold}' - skipping.")

    # Save output pkl
    try:
        with open(output_pkl_path, 'wb') as f:  # 'wb' for Windows binary
            pickle.dump(alldata, f)
        print(f"\n✅ MOSI BERT data saved: {output_pkl_path}")
    except Exception as e:
        print(f"Save error: {e}")
        sys.exit(1)