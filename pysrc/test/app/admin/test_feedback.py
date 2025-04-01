import unittest

from pysrc.app.admin.feedback import parse_feedback_content

# Part of preprocessed app.log file
LOG = """
[2021-10-12 20:25:36,219: INFO] /process regular search addr:0.0.0.0 args:{"query": "foo", "jobid": "b4450a11"}
[2020-10-24 11:31:23,312: INFO] Feedback {"key": "feedback-overview", "value": "1", "jobid": "b4450a11"}
[2021-10-12 20:25:36,219: INFO] /process regular search addr:0.0.0.0 args:{"query": "foo", "jobid": "840769eb"}
[2020-11-18 23:50:53,601: INFO] Feedback {"key": "feedback-overview", "value": "1", "jobid": "840769eb"}
[2020-11-18 23:50:53,601: INFO] Feedback {"key": "feedback-overview", "value": "0", "jobid": "840769eb"}
[2020-11-18 23:50:53,601: INFO] Feedback {"key": "cancel:feedback-overview", "value": "0", "jobid": "840769eb"}
[2020-11-18 23:50:53,601: INFO] Feedback {"key": "feedback-overview", "value": "-1", "jobid": "840769eb"}
[2021-10-12 20:25:36,219: INFO] /process regular search addr:0.0.0.0 args:{"query": "foo", "jobid": "0d16a389"}
[2020-11-23 22:17:27,340: INFO] Feedback {"key": "feedback-highlights", "value": "1", "jobid": "0d16a389"}
[2020-11-23 22:18:01,109: INFO] Feedback {"key": "feedback-network", "value": "-1", "jobid": "0d16a389"}
[2020-11-23 22:18:04,913: INFO] Feedback {"key": "cancel:feedback-network", "value": "-1", "jobid": "0d16a389"}
[2021-10-12 20:25:36,219: INFO] /process regular search addr:0.0.0.0 args:{"query": "foo", "jobid": "b607c1bd"}
"""

EXPECTED_FEEDBACK = {
    'recent': 50,
    'emotions': [('2020-11-23 22:18:04', 'network', 'Yes', 'foo'),
                 ('2020-11-23 22:18:01', 'network', 'No', 'foo'),
                 ('2020-11-23 22:17:27', 'highlights', 'Yes', 'foo'),
                 ('2020-11-18 23:50:53', 'overview', 'No', 'foo'),
                 ('2020-11-18 23:50:53', 'overview', 'Meh', 'foo'),
                 ('2020-11-18 23:50:53', 'overview', 'Meh', 'foo'),
                 ('2020-11-18 23:50:53', 'overview', 'Yes', 'foo'),
                 ('2020-10-24 11:31:23', 'overview', 'Yes', 'foo')],
    'emotions_summary': [('overview', 0.33, 3), ('highlights', 1.0, 1)]
}


class TestFeedback(unittest.TestCase):
    def test_feedback(self):
        self.assertEqual(EXPECTED_FEEDBACK, parse_feedback_content(LOG.split('\n')))


if __name__ == '__main__':
    unittest.main()
