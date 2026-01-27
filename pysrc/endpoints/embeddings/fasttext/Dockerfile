FROM biolabs/pubtrends

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

USER user

ENV PYTHONPATH=/home/user/pubtrends

# Copy application files
COPY --chown=user:pubtrends pysrc /home/user/pubtrends/pysrc/
COPY --chown=user:pubtrends scripts /home/user/pubtrends/scripts/

WORKDIR /home/user/pubtrends

EXPOSE 5001

CMD /bin/bash ~/pubtrends/scripts/nlp.sh \
   && gunicorn --bind 0.0.0.0:5001 --workers 1 --threads 1 --worker-class=gthread \
       --limit-request-line 0 --timeout=120 \
       --log-level=info --log-file=/logs/fasttext_gunicorn.log \
       --preload "pysrc.endpoints.embeddings.fasttext.fasttext_app:get_app()"
