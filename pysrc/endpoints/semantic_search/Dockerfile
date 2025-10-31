FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

ENV EMBEDDINGS_SERVICE_URL=http://embeddings:5001

RUN source activate pubtrends \
   && pip install faiss-cpu


COPY pysrc /home/user/pubtrends/pysrc/

USER root
RUN chown -R user:pubtrends /home/user/pubtrends/pysrc

USER user
WORKDIR /home/user/pubtrends

EXPOSE 5002

CMD source activate pubtrends \
   && export PYTHONPATH=/home/user/pubtrends \
   && gunicorn --bind 0.0.0.0:5002 --workers 1 --threads 1 --worker-class=gthread \
       --limit-request-line 0 --timeout=120 \
       --log-level=info --log-file=/logs/semantic_search_gunicorn.log \
       --preload "pysrc.endpoints.semantic_search.semantic_search_app:get_app()"
