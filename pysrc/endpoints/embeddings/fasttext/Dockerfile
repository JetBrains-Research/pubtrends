FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

COPY pysrc /home/user/pubtrends/pysrc/
COPY scripts /home/user/pubtrends/scripts/

USER root
RUN chown -R user:pubtrends /home/user/pubtrends/pysrc
RUN chown -R user:pubtrends /home/user/pubtrends/scripts

USER user
WORKDIR /home/user/pubtrends

EXPOSE 5001

CMD source activate pubtrends \
   && /bin/bash ~/pubtrends/scripts/nlp.sh \
   && export PYTHONPATH=/home/user/pubtrends \
   && gunicorn --bind 0.0.0.0:5001 --workers 1 --threads 1 --worker-class=gthread \
       --limit-request-line 0 --timeout=120 \
       --log-level=info --log-file=/logs/fasttext_gunicorn.log \
       --preload "pysrc.endpoints.embeddings.fasttext.fasttext_app:get_app()"
