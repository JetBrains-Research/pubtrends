# This is the main file for celery configuration, all the tasks should be registered here

from pysrc.celery.tasks_main import *

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)