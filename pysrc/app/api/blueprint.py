from flask import Blueprint, request, current_app

from pysrc.app.reports import load_or_save_result_data
from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.celery.tasks_main import analyze_search_terms, analyze_semantic_search, analyze_id_list
from pysrc.papers.utils import SORT_MOST_CITED


# API blueprint
api_bp = Blueprint('api_bp', __name__, url_prefix='/api')


def _log_request(r):
    try:
        # Mirror original logging format
        return f'addr:{r.remote_addr} args:{r.args.to_dict(flat=True)}'
    except Exception:
        return 'addr:unknown args:{}'.format(dict())


@api_bp.route('/search_terms', methods=['POST'])
def search_terms_api():
    logger = current_app.logger
    logger.info(f'/search_terms {_log_request(request)}')
    query = request.form.get('query')
    try:
        if query:
            job = analyze_search_terms.delay(
                'Pubmed', query=query, limit=1000, sort=SORT_MOST_CITED,
                noreviews=True, min_year=None, max_year=None,
                topics=10,
                test=False
            )
            return {'success': True, 'jobid': job.id}
        logger.error(f'/search_terms error {_log_request(request)}')
        return {'success': False, 'jobid': None}
    except Exception as e:
        logger.exception(f'/search_terms exception {e}')
        return {'success': False, 'jobid': None}, 500


@api_bp.route('/semantic_search', methods=['POST'])
def semantic_search_api():
    logger = current_app.logger
    logger.info(f'/semantic_search {_log_request(request)}')
    query = request.form.get('query')
    try:
        if query:
            job = analyze_semantic_search.delay(
                'Pubmed', query=query, limit=1000, noreviews=True, topics=10, test=False
            )
            return {'success': True, 'jobid': job.id}
        logger.error(f'/semantic_search error {_log_request(request)}')
        return {'success': False, 'jobid': None}
    except Exception as e:
        logger.exception(f'/semantic_search exception {e}')
        return {'success': False, 'jobid': None}, 500


@api_bp.route('/analyse_ids', methods=['POST'])
def analyse_ids_api():
    logger = current_app.logger
    logger.info(f'/analyse_ids {_log_request(request)}')
    query = request.form.get('query')
    ids_raw = request.form.get('ids')
    job_id = request.form.get('job_id')
    try:
        ids = ids_raw.split(',') if ids_raw else []
        if query and job_id and ids:
            analyze_id_list.apply_async(args=['Pubmed', query, ids, 10, False], task_id=job_id)
            return {'success': True, 'jobid': job_id}
        logger.error(f'/analyse_ids error {_log_request(request)}')
        return {'success': False, 'jobid': None}
    except Exception as e:
        logger.exception(f'/analyse_ids exception {e}')
        return {'success': False, 'jobid': None}, 500


@api_bp.route('/check_status/<jobid>', methods=['GET'])
def check_status_api(jobid):
    logger = current_app.logger
    logger.info(f'/check_status {_log_request(request)}')
    try:
        job = pubtrends_celery.AsyncResult(jobid)
        if job.state == 'PENDING':
            return {'status': 'pending'}, 200
        elif job.state == 'SUCCESS':
            return {'status': 'success'}, 200
        elif job.state == 'FAILURE':
            return {'status': 'failed'}, 200
        return {'status': 'unknown'}, 200
    except Exception as e:
        logger.exception(f'/check_status exception {e}')
        return {'status': 'error'}, 500


@api_bp.route('/get_result', methods=['GET'])
def get_result_api():
    logger = current_app.logger
    logger.info(f'/get_result {_log_request(request)}')
    jobid = request.args.get('jobid')
    query = request.args.get('query')
    try:
        if jobid and query:
            data = load_or_save_result_data(pubtrends_celery, jobid, 'Pubmed', query, SORT_MOST_CITED, 1000, True, None, None)
            return data.to_json(), 200
        return {'status': 'error'}, 500
    except Exception as e:
        logger.exception(f'/get_result exception {e}')
        return {'status': 'error'}, 500
