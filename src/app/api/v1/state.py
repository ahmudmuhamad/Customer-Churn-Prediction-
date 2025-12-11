# This dictionary holds the loaded model and artifacts in memory.
# It is populated by main.py on startup (Lifespan event).
# It is read by endpoints.py during requests.

ml_resources = {}