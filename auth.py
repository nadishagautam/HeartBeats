import os 

from flask import Flask, redirect, session, url_for, request
from config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI_LOCAL, SCOPES, REDIRECT_URI_NGROK

from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler


app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(64) #change this later

cache_handler = FlaskSessionCacheHandler(session)
sp_oauth = SpotifyOAuth(
    client_id= CLIENT_ID,
    client_secret= CLIENT_SECRET,
    redirect_uri= REDIRECT_URI_LOCAL,
    scope= SCOPES,
    cache_handler= cache_handler,
    show_dialog= True
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

    playlist_tracks = sp.playlist_tracks(playlist_id, fields='items.track.id')
    track_ids = [item['track']['id'] for item in playlist_tracks['items'] if item['track']]

    audio_features = sp.audio_features(track_ids)

    features_info = [
        {
            "id": feature['id'],
            "danceability": feature['danceability'],
            "energy": feature['energy'],
            "tempo": feature['tempo'],
            "loudness": feature['loudness']
        }
        for feature in audio_features if feature
    ]

    features_html = '<br>'.join([
        f"Track ID: {info['id']}, Danceability: {info['danceability']}, Energy: {info['energy']}, Tempo: {info['tempo']}, Loudness: {info['loudness']}"
        for info in features_info
    ])

    return features_html

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)