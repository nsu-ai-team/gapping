from codecs import open
from csv import DictReader, DictWriter, field_size_limit
import gc
import os
from sys import maxsize
from numpy import asarray
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import layers
from keras.optimizers import RMSprop
from keras.models import Model, load_model
from keras import backend as K
from sklearn.metrics import f1_score
from keras_bert.loader import load_trained_model_from_checkpoint
from keras.utils.generic_utils import get_custom_objects
from keras_bert import gelu
from keras_bert.layers import TokenEmbedding, Extract, Masked
from keras_pos_embd import PositionEmbedding
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras_position_wise_feed_forward import FeedForward
from keras_pos_embd import TrigPosEmbedding
from keras_embed_sim import EmbeddingRet, EmbeddingSim
import tokenization


class GlobalMaxPooling1D_masked(layers.GlobalMaxPooling1D):
    def __init__(self, data_format='channels_last', **kwargs):
        super(GlobalMaxPooling1D_masked, self).__init__(data_format, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        steps_axis = 1 if self.data_format == 'channels_last' else 2
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            input_shape = K.int_shape(inputs)
            broadcast_shape = [-1, input_shape[steps_axis], 1]
            mask = K.reshape(mask, broadcast_shape)
            inputs *= mask
            return K.max(inputs, axis=steps_axis)
        else:
            return K.max(inputs, axis=steps_axis)

    def compute_mask(self, inputs, mask=None):
        return None


def get_data_for_training(csv):

    field_size_limit(maxsize)
    n_samples = 0
    with open(csv, encoding="utf8") as data:
        reader = DictReader(data, delimiter='\t')
        for _ in reader:
            n_samples += 1
    print('Number of samples is {0}.'.format(n_samples))
    gc.collect()
    inp = np.zeros(shape=(n_samples, 512), dtype=np.int32)
    seg = np.zeros(shape=(n_samples, 512), dtype=np.int32)
    out0 = np.zeros(shape=(n_samples, 1), dtype=np.int32)
    out1 = np.zeros(shape=(n_samples, 512, 1), dtype=np.int32)
    out2 = np.zeros(shape=(n_samples, 512, 1), dtype=np.int32)
    out3 = np.zeros(shape=(n_samples, 512, 1), dtype=np.int32)
    out4 = np.zeros(shape=(n_samples, 512, 1), dtype=np.int32)
    out5 = np.zeros(shape=(n_samples, 512, 1), dtype=np.int32)
    out6 = np.zeros(shape=(n_samples, 512, 1), dtype=np.int32)
    sample_idx = 0
    with open(csv, encoding="utf8") as data:
        reader = DictReader(data, delimiter='\t')
        for row in reader:
            text = []
            for num in row["text"][1:-1].split(", "): text.append(int(num))
            inp[sample_idx] = asarray(text, dtype=np.int32)
            del text
            out0[sample_idx][0] = float(row["class"])
            out1[sample_idx] = generate_labels("V", row, row["class"])
            out2[sample_idx] = generate_labels("cV", row, row["class"])
            out3[sample_idx] = generate_labels("R1", row, row["class"])
            out4[sample_idx] = generate_labels("R2", row, row["class"])
            out5[sample_idx] = generate_labels("cR1", row, row["class"])
            out6[sample_idx] = generate_labels("cR2", row, row["class"])
            sample_idx += 1
            del row
    gc.collect()
    return [inp, seg], [out0, out1, out2, out3, out4, out5, out6]


def get_data_for_testing(csv):

    field_size_limit(maxsize)
    n_samples = 0
    texts = []
    with open(csv, encoding="utf8") as data:
        reader = DictReader(data, delimiter='\t')
        for _ in reader:
            n_samples += 1
    print('Number of samples is {0}.'.format(n_samples))
    gc.collect()
    inp = np.zeros(shape=(n_samples, 512), dtype=np.int32)
    seg = np.zeros(shape=(n_samples, 512), dtype=np.int32)
    sample_idx = 0
    with open(csv, encoding="utf8") as data:
        reader = DictReader(data, delimiter='\t')
        tokenizer = tokenization.FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=True)
        for row in reader:
            texts.append(row["text"])
            text = tokenizer.tokenize(row["text"])[:510]
            line = tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
            line += [0] * (512 - len(line))
            text = []
            for num in line: text.append(int(num))
            inp[sample_idx] = asarray(text, dtype=np.int32)
            del text
            sample_idx += 1
            del row
    gc.collect()
    print('All data for final testing have been loaded...')
    return [inp, seg], texts


def generate_labels(label, data, has_gap):

    out = np.zeros(shape=(512, 1), dtype=np.int32)
    if bool(has_gap):
        for ind in data[label].split(" "):
            if data[label] != "":
                start = int(ind.split(":")[0])
                end = int(ind.split(":")[1])
                if start < end:
                    for i in range(start, end): out[i + 1][0] = 1
                else:
                    out[start + 1][0] = 1
    return out


def symbol_wize(y_true1, y_pred1):

    y_true = K.round(K.clip(y_true1, 0, 1))
    y_pred = K.round(K.clip(y_pred1, 0, 1))
    true_pos = K.sum(y_true * y_pred)
    false_neg = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    false_pos = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    precision = true_pos / (true_pos + false_pos + K.epsilon())
    recall = true_pos / (true_pos + false_neg + K.epsilon())
    return 2.0 * precision * recall / (precision + recall + K.epsilon())


def f1(true, pred):

    true_positives = K.sum(K.round(K.clip(true * pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    predicted_positives = K.sum(K.round(K.clip(pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return 2.0 * precision * recall / (precision + recall + K.epsilon())


def train_full(model_path="data/model.h5", data="data/prepared.csv", nb_epoch=5, batch_size=16):

    get_custom_objects().update({'TokenEmbedding': TokenEmbedding, 'Extract': Extract, 'Masked': Masked,
                                 'PositionEmbedding': PositionEmbedding, 'LayerNormalization': LayerNormalization,
                                 'MultiHeadAttention': MultiHeadAttention, 'FeedForward': FeedForward,
                                 'TrigPosEmbedding': TrigPosEmbedding, 'EmbeddingRet': EmbeddingRet, 'gelu': gelu,
                                 'EmbeddingSim': EmbeddingSim, 'GlobalMaxPooling1D_masked': GlobalMaxPooling1D_masked,
                                 'f1': f1, 'symbol_wize': symbol_wize})
    if os.path.isfile(model_path):
        return load_model(model_path)
    train_data, train_labels = get_data_for_training(data)
    print('')
    print('Data for training have been loaded...')
    print('')
    gc.collect()
    for idx in range(3):
        print('')
        print('train_data[0]', train_data[0][idx])
        print('train_data[1]', train_data[1][idx])
        print('')
        print('train_labels[0]', train_labels[0][idx])
        print('train_labels[1]', train_labels[1][idx])
        print('train_labels[2]', train_labels[2][idx])
        print('train_labels[3]', train_labels[3][idx])
        print('train_labels[4]', train_labels[4][idx])
        print('train_labels[5]', train_labels[5][idx])
        print('train_labels[6]', train_labels[6][idx])
        print('')
    print('')
    model = load_trained_model_from_checkpoint("./multi_cased_L-12_H-768_A-12/bert_config.json",
                                               "./multi_cased_L-12_H-768_A-12/bert_model.ckpt",
                                               training=False)
    for cur in model.layers:
        if cur.name.startswith('Encoder-1') and (not cur.name.startswith('Encoder-1-')):
            cur.trainable = True
    x = model.output
    binary_pred = layers.Dense(1, name="binary", activation='sigmoid')(
        GlobalMaxPooling1D_masked(name="pooling")(x)
    )
    V_pred = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name='V')(x)
    cV_pred = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name='cV')(x)
    R1_pred = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name='R1')(x)
    R2_pred = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name='R2')(x)
    cR1_pred = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name='cR1')(x)
    cR2_pred = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name='cR2')(x)
    model = Model(model.input, [binary_pred, V_pred, cV_pred, R1_pred, R2_pred, cR1_pred, cR2_pred])
    model.compile(loss={"binary": "binary_crossentropy",
                        "V": "binary_crossentropy",
                        "cV": "binary_crossentropy",
                        "R1": "binary_crossentropy",
                        "R2": "binary_crossentropy",
                        "cR1": "binary_crossentropy",
                        "cR2": "binary_crossentropy"},
                  loss_weights={"binary": 10.0,
                                "V": 1.0,
                                "cV": 1.0,
                                "R1": 1.0,
                                "R2": 1.0,
                                "cR1": 1.0,
                                "cR2": 1.0},
                  optimizer=RMSprop(lr=1e-5),
                  metrics={"binary": f1, "V": symbol_wize, "cV": symbol_wize, "R1": symbol_wize, "R2": symbol_wize,
                           "cR1": symbol_wize, "cR2": symbol_wize})
    model.summary()
    indices = np.arange(0, len(train_data[0]), 1, dtype=np.int32)
    np.random.shuffle(indices)
    model.fit(
        [train_data[0][indices], train_data[1][indices]],
        {"binary": train_labels[0][indices], "V": train_labels[1][indices], "cV": train_labels[2][indices],
         "R1": train_labels[3][indices], "R2": train_labels[4][indices], "cR1": train_labels[5][indices],
         "cR2": train_labels[6][indices]},
        epochs=nb_epoch, batch_size=batch_size, shuffle=True, validation_split=0.1, verbose=2,
        callbacks=[ModelCheckpoint(filepath=model_path, save_best_only=True, verbose=True)]
    )

    return model
        

def test_full(model, data="data/dev_prepared.csv"):
    test_data, true_labels = get_data_for_training(data)
    gc.collect()
    print('')
    print('Data for testing have been loaded...')
    print('')
    pred_binary, pred_V, pred_cV, pred_R1, pred_R2, pred_cR1, pred_cR2 = model.predict(test_data)
    binary_pred = np.array([1 if cur[0] >= 0.5 else 0 for cur in pred_binary], dtype=np.int32)
    binary_true = np.array([1 if cur[0] >= 0.5 else 0 for cur in true_labels[0]], dtype=np.int32)
    
    return f1_score(binary_true, binary_pred), \
           K.eval(symbol_wize(K.variable(value=true_labels[1]), K.variable(value=pred_V))), \
           K.eval(symbol_wize(K.variable(value=true_labels[2]), K.variable(value=pred_cV))), \
           K.eval(symbol_wize(K.variable(value=true_labels[3]), K.variable(value=pred_R1))), \
           K.eval(symbol_wize(K.variable(value=true_labels[4]), K.variable(value=pred_R2))), \
           K.eval(symbol_wize(K.variable(value=true_labels[5]), K.variable(value=pred_cR1))), \
           K.eval(symbol_wize(K.variable(value=true_labels[6]), K.variable(value=pred_cR2)))


def detect_bounds_of_tokens(text, tokenizer):
    tokens = tokenizer.tokenize(text)[:510]
    prepared_text = text.lower().replace('й', 'и').replace('ё', 'е')
    bounds = [None]
    start_pos = 0
    for idx, val in enumerate(tokens):
        if val.startswith('##'):
            found_pos = prepared_text[start_pos:].find(val[2:])
            assert found_pos >= 0, text + ' / ' + str(tokens) + ' / ' + str(bounds) + ' / ' + ' / ' + val + ' / ' + \
                                   str(start_pos)
            bounds.append((start_pos + found_pos, start_pos + found_pos + len(val) - 2))
            start_pos = start_pos + found_pos + len(val) - 2
        elif val.startswith('[') and val.endswith(']'):
            bounds.append(None)
        else:
            found_pos = prepared_text.lower()[start_pos:].find(val)
            assert found_pos >= 0, text + ' / ' + str(tokens) + ' / ' + str(bounds) + ' / ' + ' / ' + val + ' / ' + \
                                   str(start_pos)
            bounds.append((start_pos + found_pos, start_pos + found_pos + len(val)))
            start_pos = start_pos + found_pos + len(val)
    bounds.append(None)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    tokens += ["[PAD]"] * (512 - len(tokens))
    bounds += [None] * (512 - len(bounds))
    return tokens, bounds


def get_answers(predictions, text, tokenizer):
    final, bounds = detect_bounds_of_tokens(text, tokenizer)
    indices = []
    start_pos = -1
    end_pos = -1
    cur_start_pos = 0
    cur_end_pos = 0
    for idx in range(len(final)):
        if final[idx] == '[PAD]':
            break
        if bounds[idx] is not None:
            cur_start_pos = bounds[idx][0]
            cur_end_pos = bounds[idx][1]
        if predictions[idx][0] >= 0.5:
            if start_pos < 0:
                start_pos = cur_start_pos
            end_pos = cur_end_pos
        else:
            if start_pos >= 0:
                indices.append((start_pos, end_pos))
                start_pos = -1
                end_pos = -1
    if start_pos >= 0:
        indices.append((start_pos, len(text)))
    if len(indices) == 0:
        return ''
    return ' '.join('{0}:{1}'.format(cur[0], cur[1]) for cur in indices)


def get_answers_for_V(predictions, text, tokenizer):
    final, bounds = detect_bounds_of_tokens(text, tokenizer)
    indices = []
    start_pos = 0
    for idx in range(len(final)):
        if final[idx] == '[PAD]':
            break
        if bounds[idx] is not None:
            start_pos = bounds[idx][0]
        if predictions[idx][0] >= 0.5:
            if len(indices) > 0:
                if indices[-1][1] < start_pos:
                    indices.append((start_pos, start_pos))
            else:
                indices.append((start_pos, start_pos))
    if len(indices) == 0:
        return ''
    return ' '.join('{0}:{1}'.format(cur[0], cur[1]) for cur in indices)
    

def save(out, data):                
    with open(data, "w", encoding="utf8") as new:            
        writer = DictWriter(new, delimiter="\t", fieldnames=["text", "class", "cV", "cR1", "cR2", "V", "R1", "R2"])
        writer.writeheader()
        for i in out:
            writer.writerow(i)

def get_predictions(model, data, inp, batch_size=16):
    tokenizer = tokenization.FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=True)
    predictions = []
    pred_binary, pred_V, pred_cV, pred_R1, pred_R2, pred_cR1, pred_cR2 = model.predict(data, batch_size=batch_size)
    print('All labels have been predicted...')
    a = 0
    for b, V, cV, R1, R2, cR1, cR2 in zip(pred_binary, pred_V, pred_cV, pred_R1, pred_R2, pred_cR1, pred_cR2):
        line = {"cV": "", "cR1": "", "cR2": "", "V": "", "R1": "", "R2": ""}
        line["class"] = 1 if b[0] >= 0.5 else 0
        # print(str(20904 - a) + " samples left")
        line["text"] = inp[a]
        if line["class"] == 1:
            line["V"] = get_answers_for_V(V, str(inp[a]), tokenizer)
            line["cV"] = get_answers(cV, str(inp[a]), tokenizer)
            line["R1"] = get_answers(R1, str(inp[a]), tokenizer)
            line["R2"] = get_answers(R2, str(inp[a]), tokenizer)
            line["cR1"] = get_answers(cR1, str(inp[a]), tokenizer)
            line["cR2"] = get_answers(cR2, str(inp[a]), tokenizer)
        predictions.append(line)
        del line
        a += 1
    return predictions

def main():
    model = train_full()
    res = test_full(model)
    print(res)
    model = load_model(
        "data/model.h5",
        custom_objects={'TokenEmbedding': TokenEmbedding, 'Extract': Extract, 'Masked': Masked,
                        'PositionEmbedding': PositionEmbedding, 'LayerNormalization': LayerNormalization,
                        'MultiHeadAttention': MultiHeadAttention, 'FeedForward': FeedForward,
                        'TrigPosEmbedding': TrigPosEmbedding, 'EmbeddingRet': EmbeddingRet, 'gelu': gelu,
                        'EmbeddingSim': EmbeddingSim, 'GlobalMaxPooling1D_masked': GlobalMaxPooling1D_masked,
                        'f1': f1, 'symbol_wize': symbol_wize}
    )
    test_data, texts = get_data_for_testing("data/test.csv")
    predictions = get_predictions(model, test_data, texts)
    save(predictions, "data/filled_test.csv")


if __name__ == "__main__":
    main()
