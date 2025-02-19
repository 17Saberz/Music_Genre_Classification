import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import os
from pydub import AudioSegment

#Function
def load_model():
    model = tf.keras.models.load_model("./Trained_model.h5")
    return model
    
# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(210, 210)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
                    # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
                    # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
                    # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
                #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)  # Model prediction probabilities

    # Get top 3 predictions based on highest confidence scores per sample
    top_3_indices = np.argsort(y_pred, axis=1)[:, -3:]  # Get top 3 indices per sample
    top_3_probs = np.take_along_axis(y_pred, top_3_indices, axis=1)  # Get corresponding confidence scores

    # Compute the average confidence per class across all test samples
    unique_classes = np.unique(top_3_indices)
    class_confidence = {cls: [] for cls in unique_classes}

    for i, sample_indices in enumerate(top_3_indices):
        for j, cls in enumerate(sample_indices):
            class_confidence[cls].append(top_3_probs[i, j])

    # Compute mean confidence for each class
    avg_confidence = {cls: np.mean(class_confidence[cls]) for cls in class_confidence}

    # Sort classes by highest average confidence
    sorted_classes = sorted(avg_confidence.items(), key=lambda x: x[1], reverse=True)[:3]

    # Return the top 3 classes with their confidence scores
    return sorted_classes  # List of tuples (class, confidence)


#Stramlit UI
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox("Select Page", ["Home","About Project", "Prediction"])

if(app_mode=="Home"):
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #11101f;
            color: white;
        }
        h1, h2, h3 {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h2 style='color: white;'>Welcome to the,</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white;'>Music Genre Classification System!</h2>", unsafe_allow_html=True)
    
    image_path = "./Bg_music.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("<p class='small-text'>This mini project is for 241-201</p>", unsafe_allow_html=True)
    
#About Project
elif(app_mode=="About Project"):
    st.markdown("""
                ### About Project
                Experts have long sought to understand sound and the unique characteristics that distinguish one song from another. They explore ways to visualize sound and identify the factors that make one tone different from another.
                
                This data hopefully can give the opportunity to do just that.
                
                ### About Dataset
                #### Content
                1. **genres original** - A collection of 10 genres with 100 audi files each, all having a lenght of 30 seconds (From GTZAN)
                2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, pop, reggae, rock
                """)
    
#Prediction
elif(app_mode=="Prediction"):
    st.header("Model Prediction")

    # Upload audio file
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

    if audio_file is not None:
        # Define folder to save uploaded files
        upload_folder = "music_genre"
        os.makedirs(upload_folder, exist_ok=True)  # Ensure the folder exists

        # Get the original file extension
        original_extension = audio_file.name.split(".")[-1]

        # Define the target file path (convert everything to .mp3)
        filepath = os.path.join(upload_folder, os.path.splitext(audio_file.name)[0] + ".mp3")

        # Read the uploaded file
        with open(filepath, "wb") as f:
            f.write(audio_file.getbuffer())

        # Convert file if it's not already MP3
        if original_extension.lower() != "mp3":
            st.warning(f"Converting {original_extension.upper()} to MP3...")

            # Load audio using pydub
            audio = AudioSegment.from_file(audio_file, format=original_extension)

            # Export as MP3
            audio.export(filepath, format="mp3")

            st.success(f"File converted and saved as: {filepath}")

        else:
            st.success(f"File successfully uploaded as: {filepath}")

    # Play the converted MP3 file
    if st.button("Play Audio"):
        st.audio(filepath)
        
    #Predict Button
    if(st.button("Predict")):
        with st.spinner("Please wait.."):
            X_test = load_and_preprocess_data(filepath)
            classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
            c_index = model_prediction(X_test=X_test)
            
            st.subheader("Top 3 Predicted Genres:")

            for index, confidence in c_index:
                st.markdown(f"🎵 **{classes[index]}** - Confidence: **{confidence:.2%}**")

