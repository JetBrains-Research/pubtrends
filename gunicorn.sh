# Script to launch gunicorn with optional SSL if key files present.
# Use it in CMD file, because Docker doesn't support conditional CMD.

# NOTE: Use --limit-request-line to support huge lists of papers

if [[ -f /ssl/privkey.pem ]]; then
  echo "SSL keys found. Starting gunicorn with SSL support."
  gunicorn --bind 0.0.0.0:8888 --workers 5 --limit-request-line 0 \
    --timeout=120 --log-level=info --log-file=/logs/gunicorn.log "pysrc.app.app:get_app()" \
    --keyfile=/ssl/privkey.pem --certfile=/ssl/cert.pem --ca-certs=/ssl/chain.pem
else
  echo "No SSL keys found. Starting gunicorn without SSL support."
  gunicorn --bind 0.0.0.0:8888 --workers 5 --limit-request-line 0 \
    --timeout=120 --log-level=info --log-file=/logs/gunicorn.log "pysrc.app.app:get_app()"
fi
