#!/bin/bash
PORT=${PORT:-5005}
rasa run --port $PORT --enable-api --cors "*" --host 0.0.0.0
