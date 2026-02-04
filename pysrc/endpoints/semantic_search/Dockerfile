FROM biolabs/pubtrends

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

USER user

ENV EMBEDDINGS_SERVICE_URL=http://embeddings:5001 \
    PYTHONPATH=/home/user/pubtrends

# Install additional dependencies (use absolute path since ENV PATH may not expand correctly)
RUN /home/user/.local/bin/uv pip install --no-cache faiss-cpu \
   && rm -rf /home/user/.cache

# Copy application files
COPY --chown=user:pubtrends pysrc /home/user/pubtrends/pysrc/
COPY --chown=user:pubtrends scripts /home/user/pubtrends/scripts/

WORKDIR /home/user/pubtrends

EXPOSE 5002

CMD gunicorn --bind 0.0.0.0:5002 --workers 1 --threads 1 --worker-class=gthread \
       --limit-request-line 0 --timeout=600 \
       --log-level=info --log-file=/logs/semantic_search_gunicorn.log \
       --preload "pysrc.endpoints.semantic_search.semantic_search_app:get_app()"
