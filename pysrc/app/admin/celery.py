def prepare_celery_data(app):
    inspect = app.control.inspect()
    return dict(
        active=pp(list(inspect.active().items())[0][1]),
        scheduled=pp(list(inspect.reserved().items())[0][1]),
        revoked=('\n'.join(list(inspect.revoked().items())[0][1])))


def pp(tasks):
    return '\n'.join(
        f"{t['id']} {t['name']} {','.join(str(a) for a in t['args'])} "
        f"{','.join(str(k) + ':' + str(v) for k, v in t['kwargs'].items())}"
        for t in tasks)
