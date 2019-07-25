FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

ARG PUBTRENDS_VERSION

USER user

# Updated pubtrends conda env
RUN source activate pubtrends && conda install gunicorn && pip install redis && source deactivate

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

ENV HOST 0.0.0.0
ENV PORT 8000
ENV DEBUG true

# expose the app port
EXPOSE 8000

COPY . /home/user/
WORKDIR /home/user/

# Set version info
USER root 
RUN chown -R user /home/user/models
USER user
RUN find models/app/templates/ -type f -name '*.html' | xargs -I {} sed "s#VERSION#$PUBTRENDS_VERSION#g" -i {}

# run the app server
CMD echo $PUBTRENDS_VERSION \
    && source activate pubtrends && gunicorn --bind 0.0.0.0:8000 --workers 3 "models.app.app:get_app()"
