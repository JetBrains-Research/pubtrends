FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

ENV EMBEDDINGS_SERVICE_URL=http://embeddings:5001

RUN source activate pubtrends \
   && pip install faiss-cpu


RUN mkdir -p /home/user/pubtrends/pysrc/semantic_search
COPY pysrc/endpoints/semantic_search /home/user/pubtrends/pysrc/endpoints/semantic_search
COPY pysrc/papers /home/user/pubtrends/pysrc/papers
COPY pysrc/services /home/user/pubtrends/pysrc/services
COPY pysrc/config.py /home/user/pubtrends/pysrc/

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
