FROM biolabs/pubtrends

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

USER user

# Workaround for https://github.com/JetBrains-Research/pubtrends/issues/332
# OMP: Error #13: Assertion failure at kmp_affinity.cpp(981)
# Cannot build biolabs/pubtrends with updated environment.yaml, installing manually
RUN source activate pubtrends \
    && mamba install flask-caching=2.3.1 -c conda-forge

# Download all the nltk and spacy resources
RUN source activate pubtrends \
    && python -m nltk.downloader averaged_perceptron_tagger averaged_perceptron_tagger_eng \
           punkt punkt_tab stopwords wordnet omw-1.4 \
           && python -m spacy download en_core_web_sm

# Configure URLs for between containers communication
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://redis:6379/0
ENV C_FORCE_ROOT=true
ENV FASTTEXT_URL=http://fasttext:5001


COPY . /home/user/
WORKDIR /home/user/

# Launch celery workers
ENTRYPOINT source activate pubtrends \
    && celery --app pysrc.celery.tasks worker -c 4 --loglevel=info --logfile=/logs/celery.log