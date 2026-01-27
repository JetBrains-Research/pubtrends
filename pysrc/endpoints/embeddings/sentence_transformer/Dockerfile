FROM biolabs/pubtrends

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

USER user

ENV HF_HOME=/sentence-transformers \
    PYTHONPATH=/home/user/pubtrends

# Install additional dependencies (use absolute path since ENV PATH may not expand correctly)
RUN /home/user/.local/bin/uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu \
   && /home/user/.local/bin/uv pip install --no-cache sentence-transformers faiss-cpu \
   && rm -rf /home/user/.cache

# Copy application files
COPY --chown=user:pubtrends pysrc /home/user/pubtrends/pysrc/
COPY --chown=user:pubtrends scripts /home/user/pubtrends/scripts/

WORKDIR /home/user/pubtrends

EXPOSE 5001

CMD /bin/bash ~/pubtrends/scripts/nlp.sh \
   && gunicorn --bind 0.0.0.0:5001 --workers 1 --threads 1 --worker-class=gthread \
       --limit-request-line 0 --timeout=120 \
       --log-level=info --log-file=/logs/sentence_transformer_gunicorn.log \
       --preload "pysrc.endpoints.embeddings.sentence_transformer.sentence_transformer_app:get_app()"
