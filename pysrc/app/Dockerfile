FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0

# Expose the app port, don't use default HTTP 80 or HTTPS 443
EXPOSE 8888

COPY . /home/user/
WORKDIR /home/user/

CMD source activate pubtrends; bash gunicorn.sh
