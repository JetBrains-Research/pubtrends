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

# See Update embeddings section in the README.md