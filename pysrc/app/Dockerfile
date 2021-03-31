FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0

# Run the app server with SSL certificates
RUN if [[ -f /ssl/privkey.pem ]]; then \
        echo "Certificate file found!" \
        && SSL_ARGS="--keyfile=/ssl/privkey.pem --certfile=/ssl/cert.pem --ca-certs=/ssl/chain.pem"; \
    else SSL_ARGS=""; fi;

# Expose the app port, don't use default HTTP 80 or HTTPS 443
EXPOSE 8888

COPY . /home/user/
WORKDIR /home/user/

# Use --limit-request-line to support huge lists of papers
CMD source activate pubtrends \
    && gunicorn --bind 0.0.0.0:8888 --workers 5 --limit-request-line 0 \
        --log-level=info --log-file=/logs/gunicorn.log "pysrc.app.app:get_app()" ${SSL_ARGS};
