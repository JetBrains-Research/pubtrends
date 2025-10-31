FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

ENV HF_HOME=/sentence-transformers

RUN source activate pubtrends \
   && pip install torch --index-url https://download.pytorch.org/whl/cpu \
   && pip install sentence-transformers faiss-cpu


COPY pysrc /home/user/pubtrends/pysrc/

USER root
RUN chown -R user:pubtrends /home/user/pubtrends/pysrc

USER user
WORKDIR /home/user/pubtrends

EXPOSE 5001

# See Update embeddings section in the README.md