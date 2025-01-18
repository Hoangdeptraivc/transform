import tensorflow as tf
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    LayerNormalization,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
import os
import re

# 1. CUSTOM PREPROCESSING
def custom_standardization(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r'[^a-z0-9\s\[\]]', '')
    return text

# 2. LOAD AND PREPARE DATA
def load_and_prepare_data(input_file, label_file, max_samples=25000, max_length=40):
    with open(input_file, 'r', encoding='utf-8') as f:
        input_texts = f.readlines()
    with open(label_file, 'r', encoding='utf-8') as f:
        label_texts = f.readlines()

    input_texts = [f"[sos] {line.strip()} [eos]" for line in input_texts[:max_samples]]
    label_texts = [f"[sos] {line.strip()} [eos]" for line in label_texts[:max_samples]]
    return input_texts, label_texts

# 3. CREATE TEXT VECTORIZER
def create_text_vectorizer(texts, max_tokens=10000, max_len=40):
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_sequence_length=max_len,
        standardize=custom_standardization
    )
    vectorizer.adapt(texts)
    return vectorizer

# 4. DEFINE CUSTOM LAYERS
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_len, d_model)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "max_len": self.pos_encoding.shape[1],
            "d_model": self.pos_encoding.shape[2],
        })
        return config

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i//2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, max_len, d_model):
        angle_rads = self.get_angles(
            position=tf.range(max_len, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )

        # Apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class MultiHeadAttentionLayer(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        })
        return config

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttentionLayer(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            "mha": self.mha,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
        })
        return config

    def call(self, x, mask, training=False):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttentionLayer(d_model, num_heads)
        self.mha2 = MultiHeadAttentionLayer(d_model, num_heads)

        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config.update({
            "mha1": self.mha1,
            "mha2": self.mha2,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "layernorm3": self.layernorm3,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
            "dropout3": self.dropout3,
        })
        return config

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=False):
        # Masked MHA
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # MHA with encoder output
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # FFN
        ffn_output = self.ffn(out2) # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

# 5. DEFINE MASKING FUNCTIONS
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

# 6. BUILD TRANSFORMER MODEL
class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, ff_dim, input_vocab_size, target_vocab_size, max_len, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder_embedding = Embedding(input_vocab_size, d_model)
        self.decoder_embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)

        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)
        ]
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

        self.final_layer = Dense(target_vocab_size, activation='softmax')

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({
            "encoder_embedding": self.encoder_embedding,
            "decoder_embedding": self.decoder_embedding,
            "pos_encoding": self.pos_encoding,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "dropout": self.dropout,
            "final_layer": self.final_layer,
        })
        return config

    def call(self, inputs, training=False):
        input_seq, target_seq = inputs  # Unpack the inputs

        # Create masks inside the call method
        enc_padding_mask = create_padding_mask(input_seq)
        look_ahead_mask = create_look_ahead_mask(tf.shape(target_seq)[1])
        dec_padding_mask = create_padding_mask(input_seq)

        # Encoder
        enc_emb = self.encoder_embedding(input_seq)  # (batch_size, input_seq_len, d_model)
        enc_emb = self.pos_encoding(enc_emb)
        enc_emb = self.dropout(enc_emb, training=training)

        enc_output = enc_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, enc_padding_mask, training=training)

        # Decoder
        dec_emb = self.decoder_embedding(target_seq)  # (batch_size, target_seq_len, d_model)
        dec_emb = self.pos_encoding(dec_emb)
        dec_emb = self.dropout(dec_emb, training=training)

        dec_output = dec_emb
        for dec_layer in self.decoder_layers:
            dec_output, _, _ = dec_layer(dec_output, enc_output, look_ahead_mask, dec_padding_mask, training=training)

        final_output = self.final_layer(dec_output)  # (batch_size, target_seq_len, target_vocab_size)

        return final_output

# 7. INFERENCE FUNCTION
def decode_sequence(model, input_vectorizer, label_vectorizer, input_text, max_len=40):
    vocab = label_vectorizer.get_vocabulary()
    start_token = vocab.index('[sos]')
    end_token = vocab.index('[eos]')

    input_seq = input_vectorizer([f"[sos] {input_text.strip()} [eos]"])
    input_seq = tf.cast(input_seq, tf.int32)  # Ensure dtype matches

    output_seq = tf.expand_dims([start_token], 0)  # (1, 1)

    for _ in range(max_len):
        predictions = model(inputs=[input_seq, output_seq], training=False)  # (batch_size, seq_len, vocab_size)

        # Select the last word
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.argmax(predictions, axis=-1)  # (batch_size, 1)

        if predicted_id.numpy()[0][0] == end_token:
            break

        # Ensure predicted_id is int32 to match output_seq
        predicted_id = tf.cast(predicted_id, output_seq.dtype)

        # Concatenate the predicted_id to the output_seq
        output_seq = tf.concat([output_seq, predicted_id], axis=-1)

    predicted_sentence = [
        vocab[token_id] for token_id in output_seq.numpy()[0] if token_id < len(vocab)
    ]
    return ' '.join(predicted_sentence).replace('[sos]', '').replace('[eos]', '').strip()

# 8. TRAINING AND PREDICTION
# Paths
input_file = 'input_texts.txt'
label_file = 'label_texts.txt'

# Parameters
max_samples = 25000
max_length = 40
d_model = 256
num_heads = 8
ff_dim = 512
num_layers = 2
dropout_rate = 0.1
epochs = 80
batch_size = 64

# Load and prepare data
input_texts, label_texts = load_and_prepare_data(input_file, label_file, max_samples, max_length)

# Create vectorizers
input_vectorizer = create_text_vectorizer(input_texts, max_tokens=10000, max_len=max_length)
label_vectorizer = create_text_vectorizer(label_texts, max_tokens=10000, max_len=max_length)

# Vectorize data
input_data = input_vectorizer(input_texts)
input_data = tf.cast(input_data, tf.int32)  # Cast to int32

label_data = label_vectorizer(label_texts)
label_data = tf.cast(label_data, tf.int32)  # Cast to int32

# Prepare decoder input and target
label_input_data = label_data[:, :-1]
label_target_data = label_data[:, 1:]

# Vocabulary sizes
input_vocab_size = len(input_vectorizer.get_vocabulary())
target_vocab_size = len(label_vectorizer.get_vocabulary())

# Build model
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    ff_dim=ff_dim,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    max_len=max_length,
    dropout_rate=dropout_rate
)

# Compile model
transformer.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
transformer.fit(
    x=[input_data, label_input_data],
    y=label_target_data,
    epochs=epochs,
    batch_size=batch_size
)

# Test inference
test_input = "hi, how are you doing?"
response = decode_sequence(transformer, input_vectorizer, label_vectorizer, test_input)
print("Input:", test_input)
print("Response:", response)
