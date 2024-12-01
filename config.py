import os

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

REDIRECT_URI_LOCAL = "http://localhost:8080/callback"

SCOPES= "user-library-read user-top-read playlist-read-private playlist-read-collaborative"
