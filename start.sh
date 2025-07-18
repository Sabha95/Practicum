#!/bin/bash
PORT=${PORT:-5005}
rasa run --port $PORT --enable-api --cors "*" 