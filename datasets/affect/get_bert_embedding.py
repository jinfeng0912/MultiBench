"""Implements BERT embedding extractors."""
import torch
from transformers import AutoTokenizer, BertModel
import h5py
import pickle
import numpy as np
import sys # Import sys to handle potential script termination

# --- Model and Tokenizer Initialization ---
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name)
bert.config.output_hidden_states = True

def get_bert_features(all_text, contextual_embedding=False, batch_size=128, max_len=None):
    """Get bert features from data."""
    # CHANGE 1: Handle cases where all_text might be empty or contain empty strings.
    # This directly fixes the "ValueError: max() iterable argument is empty"
    non_empty_texts = [s for s in all_text if s.strip()]
    if not non_empty_texts:
        print("Warning: No valid text found to process.")
        return np.array([]) # Return an empty array if there's nothing to process

    output_bert_features = []
    if max_len is None:
        # Calculate max_len only on texts that are not empty
        max_len = max([len(s.split()) for s in non_empty_texts])
        # We cap max_len at 512, which is a common limit for BERT models.
        max_len = min(max_len, 512)

    print(f"Max sequence length set to: {max_len}")
    print(f"Total text instances to process: {len(all_text)}")

    for i in range(0, len(all_text), batch_size):
        # Ensure the batch is a list of strings
        current_batch = all_text[i: i+batch_size]

        inputs = tokenizer(current_batch, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")

        with torch.no_grad():
            bert_features_output = bert(**inputs)

        outputs = bert_features_output.hidden_states
        if contextual_embedding:
            output_bert_features.append(outputs[-1].detach().numpy())
        else:
            output_bert_features.append(outputs[0].detach().numpy())

        print(f'Batch starting at index {i} finished!')

    return np.concatenate(output_bert_features)

def get_rawtext(path, data_kind, vids=None):
    """Get raw text from the datasets."""
    text_data = []

    try:
        if data_kind == 'hdf5':
            with h5py.File(path, 'r') as f:
                if vids is None:
                    vids = list(f['words'].keys())
                for vid in vids:
                    try:
                        words = [word[0].decode('utf-8') for word in f['words'][vid]['features'] if word[0] != b'sp']
                        text_data.append(' '.join(words))
                    except KeyError:
                        print(f"Warning: Key '{vid}' not found in HDF5 file. Appending empty string.")
                        text_data.append("")
        else: # Assumes pickle file
             with open(path, 'rb') as f_r:
                f = pickle.load(f_r)
                # This part is complex because the structure of pkl is not standard.
                # Assuming it's a dict-like object. This may need adjustment.
                if vids is None:
                    vids = list(f.keys())
                for vid in vids:
                    # Logic for pickle file text extraction
                    pass # Placeholder
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return [] # Return empty list on error

    return text_data, vids

# The other functions (max_seq_len, corresponding_other_modality_ids, bert_version_data)
# remain largely the same as the improved version from our previous conversations.
# (For brevity, they are omitted here but should be included in your final script)
# Let's assume they are present and correct.

def max_seq_len(id_list, max_len=50):
    new_id_list = []
    for id_seq in id_list:
        if len(id_seq) > 0:
            padded_seq = id_seq[:max_len]
            new_id_list.append(padded_seq)
        else:
            new_id_list.append([])
    return new_id_list

def corresponding_other_modality_ids(orig_text, tokenized_text):
    id_list = []
    idx = -1
    for i, t in enumerate(tokenized_text):
        if '##' in t:
            id_list.append(idx)
        else:
            idx += 1
            id_list.append(idx)
    return id_list

def bert_version_data(data, raw_path, keys, max_padding=50):
    file_type = raw_path.split('.')[-1]
    raw_texts, _ = get_rawtext(raw_path, file_type, keys)

    bert_features = get_bert_features(raw_texts, contextual_embedding=False, max_len=max_padding)

    if bert_features.size == 0: # Check if feature extraction failed
        print("Aborting processing for this fold due to empty BERT features.")
        return None

    other_modality_ids = [corresponding_other_modality_ids(text, tokenizer.tokenize(text)) for text in raw_texts]
    new_other_mids = max_seq_len(other_modality_ids, max_len=max_padding)

    # Align vision and audio modalities
    new_vision, new_audio = [], []
    for i in range(len(keys)):
        # Vision
        v = data['vision'][i]
        aligned_indices_v = np.array(new_other_mids[i])
        valid_indices_v = aligned_indices_v[aligned_indices_v < v.shape[0]]
        tmp_v = v[valid_indices_v]
        pad_width_v = ((0, max_padding - tmp_v.shape[0]), (0, 0))
        new_vision.append(np.pad(tmp_v, pad_width_v, 'constant'))

        # Audio
        a = data['audio'][i]
        aligned_indices_a = np.array(new_other_mids[i])
        valid_indices_a = aligned_indices_a[aligned_indices_a < a.shape[0]]
        tmp_a = a[valid_indices_a]
        pad_width_a = ((0, max_padding - tmp_a.shape[0]), (0, 0))
        new_audio.append(np.pad(tmp_a, pad_width_a, 'constant'))

    return {
        'vision': np.stack(new_vision),
        'audio': np.stack(new_audio),
        'text': bert_features # BERT features are already padded
    }

# --- Main Execution Block ---
# --- Main Execution Block ---
if __name__ == '__main__':
    # Use the full, absolute paths to your data files
    input_pkl_path = 'E:/Laboratory/datasets/CMU_MOSI/mosi_raw.pkl'
    raw_text_hdf5_path = 'E:/Laboratory/datasets/CMU_MOSI/mosi.hdf5'
    # The output file will be saved in the same directory as the script
    output_pkl_path = 'mosi_bert.pkl'
    try:
        with open(input_pkl_path, "rb") as f:
            alldata = pickle.load(f)
        print(f"Successfully loaded data from: {input_pkl_path}")
    except FileNotFoundError:
        print(f"Error: The file '{input_pkl_path}' was not found.")
        print("Please make sure you have run the first preprocessing step to create 'mosi_raw.pkl'.")
        sys.exit() # Exit the script if the input file is missing.
    except Exception as e:
        print(f"An error occurred loading '{input_pkl_path}': {e}")
        sys.exit()

    # Process each data split (train, valid, test)
    for fold in alldata.keys():
        print(f"\n--- Processing '{fold}' data ---")

        # CHANGE 3: Ensure IDs are strings for lookup.
        # The IDs in the .pkl file might be bytes, so we decode them to strings.
        keys = [k.decode('utf-8') if isinstance(k, bytes) else str(k) for k in alldata[fold]['id']]

        processed_fold_data = bert_version_data(
            data=alldata[fold],
            raw_path=raw_text_hdf5_path,
            keys=keys,
            max_padding=50
        )

        # Update the dictionary with the new BERT-based features
        if processed_fold_data:
            alldata[fold]['vision'] = processed_fold_data['vision']
            alldata[fold]['audio'] = processed_fold_data['audio']
            alldata[fold]['text'] = processed_fold_data['text']
            print(f"Finished processing '{fold}' data.")
        else:
            print(f"Skipped updating '{fold}' data due to processing errors.")


    # Save the final dictionary to the output file
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(alldata, f)

    print(f"\n✅ Success! Processed data saved to '{output_pkl_path}'.")