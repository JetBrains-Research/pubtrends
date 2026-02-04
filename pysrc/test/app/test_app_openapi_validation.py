"""Test Flask app routes against OpenAPI specification."""
import os.path
from pathlib import Path

import pytest
import yaml
from pysrc.app.pubtrends_app import pubtrends_app


@pytest.fixture(scope="module")
def openapi_spec():
    """Load OpenAPI specification."""
    spec_path = Path(__file__).parent.parent.parent / "app" / "openapi.yaml"
    with open(spec_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def required_endpoints(openapi_spec):
    """Extract required endpoints from OpenAPI spec."""
    endpoints = []
    for path, methods_data in openapi_spec["paths"].items():
        methods = [method.upper() for method in methods_data.keys()]
        # Convert OpenAPI path params {param} to Flask style <param>
        flask_path = path.replace("{", "<").replace("}", ">")
        endpoints.append({"path": flask_path, "methods": set(methods)})
    return endpoints


@pytest.fixture(scope="module")
def actual_routes():
    """Extract all routes from the Flask app."""
    with pubtrends_app.test_request_context():
        return [
            {"path": rule.rule, "methods": rule.methods - {'HEAD', 'OPTIONS'}}
            for rule in pubtrends_app.url_map.iter_rules()
            if rule.endpoint != 'static'
        ]


def test_required_routes_exist(required_endpoints, actual_routes):
    """Validate that all OpenAPI-defined routes exist in Flask app."""
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


def test_no_undocumented_routes(required_endpoints, actual_routes):
    """Validate that Flask app has no undocumented routes."""
    required_paths = {req["path"] for req in required_endpoints}
    actual_paths = {route["path"] for route in actual_routes}

    # Filter out admin routes as they may be intentionally undocumented
    undocumented = {p for p in actual_paths - required_paths if not p.startswith('/admin')}

    if undocumented:
        print(f"\nWarning: Found {len(undocumented)} undocumented routes:")
        for path in sorted(undocumented):
            print(f"  - {path}")
