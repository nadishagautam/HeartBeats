# HeartBeats

## Setup and Run the Demo

This project demonstrates how to train a machine learning model and use it to classify a playlist of songs based on intensity levels. Follow the steps below to get started.

### 1. Set up a Virtual Environment

To create and activate a virtual environment, run the following commands in the main directory:

#### **For Windows:**
```bash
python -m venv venv
venv/Scripts/activate
```

#### **For MacOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install the Dependencies
Once the virtual environment is activated, install the required dependencies from the `requirements.txt` file (which is in the main directory):
```bash
pip install -r requirements.txt
```

### 3. Run the Demo
To train the model and classify a playlist of songs, navigate to the **`demo`** folder and run the Jupyter Notebooks. 

`train_model.ipynb` builds, trains, and saves the neural network.

`test_model.ipynb` loads the trained model and classifies a playlist of songs.

`model.py` stores the implementation of the neural network.

### Note

We would like to mention the Spotify API was used to retrieve song attributes such as tempo, energy, loudness, and valence from a User’s Spotify Playlist. This functionality is included in the code in the folder SpotifyAPI but should be noted that the API is no longer functional in this setup. It is referenced in the code but is not operational due to API accesspoint being removed by Spotify on Novemeber 27th, 2024. 

