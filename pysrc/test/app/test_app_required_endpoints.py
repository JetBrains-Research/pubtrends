import json
import os.path

import pytest
from pysrc.app.pubtrends_app import pubtrends_app  # your Flask app

# Load required endpoints from a JSON file
@pytest.fixture(scope="module")
def required_endpoints():
    requirements_path = os.path.join(os.path.dirname(__file__), "../../../pysrc/app/required_endpoints.json")
    with open(requirements_path) as f:
        return [
            {"path": ep["path"], "methods": set(ep["methods"])}
            for ep in json.load(f)
        ]

# Extract all routes from the Flask app
@pytest.fixture(scope="module")
def actual_routes():
    with pubtrends_app.test_request_context():
        return [
            {"path": rule.rule, "methods": rule.methods - {'HEAD', 'OPTIONS'}}
            for rule in pubtrends_app.url_map.iter_rules()
            if rule.endpoint != 'static'
        ]

def test_required_routes_exist(required_endpoints, actual_routes):
    errors = []

    for req in required_endpoints:
        match = next((r for r in actual_routes if r["path"] == req["path"]), None)
        if not match:
            errors.append(f"Missing path: {req['path']}")
        else:
            missing_methods = req["methods"] - match["methods"]
            if missing_methods:
                errors.append(f"Missing methods for {req['path']}: {missing_methods}")

    assert not errors, "\n" + "\n".join(errors)
