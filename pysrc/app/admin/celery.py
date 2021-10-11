import pprint


def prepare_celery_data(app):
    inspect = app.control.inspect()
    return dict(
        active=pprint.pformat(inspect.active()),
        scheduled=pprint.pformat(inspect.scheduled()),
        revoked=pprint.pformat(inspect.revoked())
    )
