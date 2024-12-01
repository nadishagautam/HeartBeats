import os 
from flask import Flask, redirect, session, url_for, request
from config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI_LOCAL, SCOPES
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(64)  # change this later

cache_handler = FlaskSessionCacheHandler(session)
sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI_LOCAL,
    scope=SCOPES,
    cache_handler=cache_handler,
    show_dialog=True
)

sp = Spotify(auth_manager=sp_oauth)

@app.route('/')
def home():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    return redirect(url_for('get_playlist_audio_features'))

@app.route('/callback')
def callback():
    sp_oauth.get_access_token(request.args['code'])
    return redirect(url_for('get_playlist_audio_features'))

@app.route('/get_playlist_audio_features')
def get_playlist_audio_features():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    
    playlist_id = '35np3i3AmrZnBbfui7wf78'

    # Get the tracks from the playlist
    playlist_tracks = sp.playlist_tracks(playlist_id, fields='items.track.id,items.track.name')

    # Extract the track IDs and names
    track_data = [(item['track']['name'], item['track']['id']) for item in playlist_tracks['items'] if item['track']]

    # Fetch the audio features for the tracks
    track_audio_features = sp.audio_features([track[1] for track in track_data])

    # Format the track names and their corresponding audio features
    track_features_info = [
        {
            "name": track[0],
            "tempo": feature['tempo'] if feature else None,
            "energy": feature['energy'] if feature else None,
            "loudness": feature['loudness'] if feature else None,
            "valence": feature['valence'] if feature else None
        }
        for track, feature in zip(track_data, track_audio_features)
    ]

    # Format the data into an HTML string
    features_html = '<br>'.join([
        f"Song: {info['name']}, Tempo: {info['tempo']}, Energy: {info['energy']}, Loudness: {info['loudness']}, Valence: {info['valence']}"
        for info in track_features_info
    ])

    return features_html

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
