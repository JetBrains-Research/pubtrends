FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

# expose the app port, don't use default HTTPS 443 here, it will be used by docker compose
EXPOSE 4433

COPY . /home/user/
WORKDIR /home/user/

# run the app server with SSL sertificates
CMD echo $PUBTRENDS_VERSION \
    && source activate pubtrends \
    && gunicorn --bind 0.0.0.0:4433 --workers 5 \
        --log-level=info --log-file=/logs/gunicorn.log "pysrc.app.app:get_app()" \
        --keyfile=/ssl/privkey.pem --certfile=/ssl/cert.pem \
        --ca-certs=/ssl/chain.pem
