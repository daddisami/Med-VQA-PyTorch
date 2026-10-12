import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, LSTM, Embedding, Multiply
from tensorflow.keras.models import Model

NUM_ANS_TOP = 15

# --- MODEL DEFINITION ---
def build_cnn_baseline():
    inp = Input(shape=(224, 224, 3))
    base = ResNet50(weights='imagenet', include_top=False, input_tensor=inp)
    # Freezing the base layers
    for layer in base.layers: layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(NUM_ANS_TOP, activation='softmax')(x)
    return Model(inp, out)


# --- VLM MODEL ---
def build_vlm_model(vocab_size, embed_matrix, MAX_Q_LEN, EMBED_DIM):
    # ResNet50, all layers frozen, without the classification layer
    in_img = Input((224,224,3))
    base = ResNet50(weights='imagenet', include_top=False, input_tensor=in_img)
    for l in base.layers:
        l.trainable = False

    # convert 3D feature map into 1D vector
    x_img = GlobalAveragePooling2D()(base.output)

    # projecting feature to a lower dimension
    x_img = Dense(512, activation='relu')(x_img)
    x_img = Dropout(0.5)(x_img)

    # input layer for tokenized question sequences
    in_txt = Input((MAX_Q_LEN,))

    # embedding layer pre-trained with Word2Vec weights ( we keep embedding static)
    emb = Embedding(vocab_size, EMBED_DIM, weights=[embed_matrix],
                    trainable=False, mask_zero=True)(in_txt)
    x_txt = LSTM(256)(emb)
    x_txt = Dropout(0.5)(x_txt)

    # multimodal fusion
    # we project both modalities to the same dimension then we use element wise multiplication to fuse them
    fusion = Multiply()([Dense(256, activation='relu')(x_img),
                         Dense(256, activation='relu')(x_txt)])
    # Post-fusion processing
    x = Dense(256, activation='relu')(fusion)
    x = Dropout(0.5)(x)

    # Output layer: Softmax activation for multi-class classification
    out = Dense(NUM_ANS_TOP, activation='softmax')(x)
    model = Model([in_img, in_txt], out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

