import logging
import re

from celery.result import AsyncResult
from flask import request, redirect, url_for, render_template_string, render_template

from pysrc.app.predefined import load_predefined_or_result_data
from pysrc.app.utils import log_request, SOMETHING_WENT_WRONG_SEARCH, ERROR_OCCURRED, MAX_QUERY_LENGTH
from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.utils import trim
from pysrc.version import VERSION
from pysrc.review.app.task import prepare_review_data_async

logger = logging.getLogger(__name__)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

REVIEW_ANALYSIS_TYPE = 'review'


def register_app_review(app):

    @app.route('/generate_review')
    def generate_review():
        logger.info(f'/generate_review {log_request(request)}')
        try:
            jobid = request.args.get('jobid')
            query = request.args.get('query')
            source = request.args.get('source')
            limit = request.args.get('limit')
            sort = request.args.get('sort')
            num_papers = request.args.get('papers_number')
            num_sents = request.args.get('sents_number')
            if jobid:
                data = load_predefined_or_result_data(jobid, pubtrends_celery)
                if data is not None:
                    job = prepare_review_data_async.delay(data, source, num_papers, num_sents)
                    return redirect(url_for('.process', analysis_type=REVIEW_ANALYSIS_TYPE, jobid=job.id,
                                            query=query, source=source, limit=limit, sort=sort))
            logger.error(f'/generate_review error {log_request(request)}')
            return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
        except Exception as e:
            logger.error(f'/generate_review error', e)
            return render_template_string(ERROR_OCCURRED), 500

    @app.route('/review')
    def review():
        logger.info(f'/review {log_request(request)}')
        try:
            jobid = request.args.get('jobid')
            query = request.args.get('query')
            source = request.args.get('source')
            limit = request.args.get('limit')
            sort = request.args.get('sort')
            if jobid:
                job = AsyncResult(jobid, app=pubtrends_celery)
                if job and job.state == 'SUCCESS':
                    review_res = job.result
                    export_name = re.sub('_{2,}', '_', re.sub('["\':,. ]', '_', f'{query}_review'.lower().strip('_')))
                    return render_template('review.html',
                                           query=trim(query, MAX_QUERY_LENGTH),
                                           source=source,
                                           limit=limit,
                                           sort=sort,
                                           version=VERSION,
                                           review_array=review_res,
                                           export_name=export_name)
            else:
                logger.error(f'/review error {log_request(request)}')
                return render_template_string(SOMETHING_WENT_WRONG_SEARCH), 400
        except Exception as e:
            logger.error(f'/review error', e)
            return render_template_string(ERROR_OCCURRED), 500
