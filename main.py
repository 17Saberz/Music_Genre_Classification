import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import yt_dlp

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
    avg_confidences = np.mean(y_pred, axis=0)  # Average confidence across chunks

    # Sort classes by highest average confidence
    sorted_classes = sorted(avg_confidence.items(), key=lambda x: x[1], reverse=True)[:3]

    # Return the top 3 classes with their confidence scores
    return sorted_classes, avg_confidences  # List of tuples (class, confidence)

# Function to download and convert YouTube audio to MP3
def download_youtube_audio(youtube_url, save_folder="music_genre"):
    os.makedirs(save_folder, exist_ok=True)  # Ensure the folder exists
    
    # Define file path
    filename = os.path.join(save_folder, "youtube_audio.mp3")
    
    # yt_dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(save_folder, 'youtube_audio.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    
    # Download audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    # Rename downloaded file to standard filename
    for file in os.listdir(save_folder):
        if file.startswith("youtube_audio") and file.endswith(".mp3"):
            os.rename(os.path.join(save_folder, file), filename)
            break
    
    return filename

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

    # Option to upload file OR enter YouTube link
    upload_option = st.radio("Select Input Method:", ["Upload Audio File", "Enter YouTube Link"])
    
    if upload_option == "Upload Audio File":
        audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "mp4"])
        
        if audio_file is not None:
            upload_folder = "music_genre"
            os.makedirs(upload_folder, exist_ok=True)
    
            # Get file extension
            original_extension = audio_file.name.split(".")[-1].lower()
    
            # Define the target MP3 file path
            filepath = os.path.join(upload_folder, os.path.splitext(audio_file.name)[0] + ".mp3")
    
            # Save the uploaded file temporarily
            temp_path = os.path.join(upload_folder, audio_file.name)
            with open(temp_path, "wb") as f:
                f.write(audio_file.getbuffer())
    
            # Convert file if it's not already MP3
            if original_extension != "mp3":
                st.warning(f"Converting {original_extension.upper()} to MP3...")
                audio = AudioSegment.from_file(temp_path, format=original_extension)
                audio.export(filepath, format="mp3")
                os.remove(temp_path)
                st.success(f"File converted and saved as: {filepath}")
            else:
                st.success(f"File successfully uploaded as: {filepath}")
    
    elif upload_option == "Enter YouTube Link":
        youtube_url = st.text_input("Enter YouTube video URL:")
        
        if youtube_url:
            st.warning("Downloading and extracting audio from YouTube...")
            filepath = download_youtube_audio(youtube_url)
            st.success(f"Audio extracted and saved as: {filepath}")
    
    # Play the audio
    if 'filepath' in locals() and st.button("Play Audio"):
        st.audio(filepath)
    
    # Predict Button
    if 'filepath' in locals() and st.button("Predict"):
        with st.spinner("Please wait.."):
            X_test = load_and_preprocess_data(filepath)
            classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            
            # Get all genre confidence scores
            model = load_model()
            y_pred = model.predict(X_test)
            avg_confidences = np.mean(y_pred, axis=0)
    
            # Display all genre predictions as text
            st.subheader("Predicted Genre Probabilities:")
            for genre, confidence in zip(classes, avg_confidences):
                st.markdown(f"🎵 **{genre}** - Confidence: **{confidence:.2%}**")
    
            # Pie Chart for All 10 Genres
            fig1, ax1 = plt.subplots()
            ax1.pie(avg_confidences, labels=classes, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax1.axis('equal')
            st.subheader("Confidence Distribution (Pie Chart)")
            st.pyplot(fig1)
    
            # Bar Chart for All 10 Genres
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.barh(classes, avg_confidences, color='skyblue')
            ax2.set_xlabel("Confidence Score")
            ax2.set_title("Confidence Scores for All Genres")
            ax2.invert_yaxis()
            st.subheader("Confidence Distribution (Bar Chart)")
            st.pyplot(fig2)