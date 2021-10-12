import json

MAX_QUERY_LENGTH = 60

SOMETHING_WENT_WRONG_SEARCH = 'Something went wrong, please <a href="/">rerun</a> your search.'
SOMETHING_WENT_WRONG_TOPIC = 'Something went wrong, please <a href="/">rerun</a> your topic analysis.'
SOMETHING_WENT_WRONG_PAPER = 'Something went wrong, please <a href="/#paper-tab">rerun</a> your paper analysis.'
ERROR_OCCURRED = "Error occurred. We're working on it. Please check back soon."


def log_request(r):
    return f'addr:{r.remote_addr} args:{json.dumps(r.args)}'
