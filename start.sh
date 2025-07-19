#!/bin/bash

echo "TESTING CLI..."

which rasa
rasa --version

PORT=${PORT:-5005}
echo "Trying command:"
echo "rasa run --port $PORT --enable-api --host 0.0.0.0"
rasa run --port 5005 --enable-api --cors "*" --debug
