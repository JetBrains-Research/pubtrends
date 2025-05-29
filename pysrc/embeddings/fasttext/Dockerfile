FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

# Download all the nltk and spacy resources
COPY nlp.sh /home/user/nlp.sh
RUN /bin/bash ~/nlp.sh

RUN mkdir -p /home/user/pubtrends/pysrc/embeddings/fasttext
COPY pysrc/embeddings/fasttext /home/user/pubtrends/pysrc/embeddings/fasttext/

USER root
RUN chown -R user:pubtrends /home/user/pubtrends/pysrc

USER user
WORKDIR /home/user/pubtrends

EXPOSE 5001

CMD source activate pubtrends \
   && export PYTHONPATH=/home/user/pubtrends \
   && gunicorn --bind 0.0.0.0:5001 --workers 1 --threads 1 --worker-class=gthread \
       --limit-request-line 0 --timeout=120 \
       --log-level=info --log-file=/logs/fasttext_gunicorn.log \
       --preload "pysrc.embeddings.fasttext.fasttext_app:get_app()"
