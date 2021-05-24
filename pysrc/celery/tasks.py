# This is the main file for celery configuration, all the tasks should be registered here

from pysrc.celery.tasks_main import *

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)
if PUBTRENDS_CONFIG.feature_review_enabled:
    # noinspection PyUnresolvedReferences
    from pysrc.review.app.task import prepare_review_data_async
