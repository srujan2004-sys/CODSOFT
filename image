import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Add

# Load pre-trained InceptionV3 model and remove the final layer
def create_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

# Extract features from images
def extract_features(image, model):
    image = tf.image.resize(image, (299, 299))
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return model.predict(image)

# Define the RNN decoder model for caption generation
def create_captioning_model(vocab_size, max_length, feature_dim=2048):
    # Image feature input
    image_input = Input(shape=(feature_dim,))
    image_features = Dense(256, activation='relu')(image_input)
    
    # Sequence input for text
    sequence_input = Input(shape=(max_length,))
    sequence_embedding = Embedding(vocab_size, 256, mask_zero=True)(sequence_input)
    sequence_features = LSTM(256)(sequence_embedding)
    
    # Combine image and sequence features
    decoder_input = Add()([image_features, sequence_features])
    decoder_output = Dense(vocab_size, activation='softmax')(decoder_input)
    
    model = Model(inputs=[image_input, sequence_input], outputs=decoder_output)
    return model

# Training function
def train_captioning_model(model, dataset, tokenizer, max_length, epochs=20):
    for epoch in range(epochs):
        for image, caption in dataset:
            # Prepare image features and caption input
            image_features = extract_features(image, feature_extractor)
            input_seq = tokenizer.texts_to_sequences([caption])[0]
            input_seq = pad_sequences([input_seq], maxlen=max_length)
            
            # Train the model
            loss = model.train_on_batch([image_features, input_seq], target_seq)

# Example usage:
vocab_size = 5000
max_length = 20
feature_extractor = create_feature_extractor()
captioning_model = create_captioning_model(vocab_size, max_length)
