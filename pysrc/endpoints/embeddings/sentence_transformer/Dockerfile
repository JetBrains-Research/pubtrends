FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

ENV HF_HOME=/sentence-transformers

RUN source activate pubtrends \
   && pip install torch --index-url https://download.pytorch.org/whl/cpu \
   && pip install sentence-transformers

RUN mkdir -p /home/user/pubtrends/pysrc/endpoints/embeddings/sentence_transformer
COPY pysrc/endpoints/embeddings/sentence_transformer /home/user/pubtrends/pysrc/endpoints/embeddings/sentence_transformer/
COPY pysrc/config.py /home/user/pubtrends/pysrc/

USER root
RUN chown -R user:pubtrends /home/user/pubtrends/pysrc

USER user
WORKDIR /home/user/pubtrends

EXPOSE 5001

CMD source activate pubtrends \
   && export PYTHONPATH=/home/user/pubtrends \
   && gunicorn --bind 0.0.0.0:5001 --workers 1 --threads 1 --worker-class=gthread \
       --limit-request-line 0 --timeout=120 \
       --log-level=info --log-file=/logs/sentence_transformer_gunicorn.log \
       --preload "pysrc.endpoints.embeddings.sentence_transformer.sentence_transformer_app:get_app()"
