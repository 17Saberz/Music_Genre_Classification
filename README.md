# Music Genre Classification

This project is a web application for music genre classification using deep learning. Users can upload audio files or provide YouTube links to classify the audio into one of ten music genres: `['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']`. The app is built using TensorFlow, Streamlit, and various audio processing libraries.

## Project Overview

The objective of this project is to analyze audio and classify it into one of ten predefined music genres. This project showcases audio processing, deep learning, and user-friendly interaction via Streamlit.

## Technologies Used

- **Backend:** TensorFlow, Librosa, Pydub
- **Frontend:** Streamlit
- **Visualization:** Matplotlib
- **Audio Processing:** Librosa, Pydub
- **Download Audio:** yt-dlp for downloading and converting YouTube audio

## Installation and Setup

To run this project locally using Anaconda, follow these steps:

1. Clone the repository:

    ```bash
    git clone <repository_url>
    ```

2. Create a new Anaconda environment:

    ```bash
    conda create --name genre_classification python=3.9
    conda activate genre_classification
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    conda install tensorflow librosa matplotlib pydub yt-dlp
    ```

4. Make sure to install FFmpeg if not installed:

    ```bash
    conda install -c conda-forge ffmpeg
    ```

## Running the Application

To start the Streamlit app:

```bash
streamlit run main.py
```

Then, open a web browser and navigate to `http://localhost:8501`.

## Usage

1. **Home Page:** Explore the home page with welcome information.
2. **About Project:** Learn more about the purpose and details of the dataset.
3. **Prediction Page:**
    - Upload an audio file (mp3, wav, ogg, mp4) or enter a YouTube link.
    - Play the uploaded audio file.
    - Get genre predictions and confidence scores.
    - Visualize the prediction results through pie charts and bar charts.

## Dataset

- The dataset consists of audio files of 10 genres: `['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']`.
- Data includes 100 audio files per genre, each of 30 seconds in length (sourced from GTZAN dataset).

## Troubleshooting

- **Problem:** Issues with `ffmpeg` not found.
  - **Solution:** Run:
  
    ```bash
    conda install -c conda-forge ffmpeg
    ```

- **Problem:** Errors while downloading audio using `yt-dlp`.
  - **Solution:** Make sure that `yt-dlp` is installed and properly configured in your environment.

## Project Details

- This project leverages machine learning to classify audio based on spectrogram analysis.
- The model is trained on the GTZAN dataset using a convolutional neural network (CNN).

---
## Loss Result
![Image](https://github.com/user-attachments/assets/25db9d5b-2ccf-41f2-a580-77ccaf975353)

## Accuracy Result
![Image](https://github.com/user-attachments/assets/31e67dc5-4400-43b8-9493-988ac0ef825c)

## Confusion Matrix
![Image](https://github.com/user-attachments/assets/82e3077f-2975-4849-813b-40d4da3a96fc)

---
