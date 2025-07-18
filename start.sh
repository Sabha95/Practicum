#!/bin/bash
echo "Starting Rasa on port $PORT..."
PORT=${PORT:-5005}
rasa run --enable-api --port $PORT --host 0.0.0.0 --cors "*" --debug
