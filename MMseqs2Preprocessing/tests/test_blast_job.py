"""Offline checks for sequential BLAST processing and its pauses."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('blast_job', Path(__file__).parents[1] / 'blastJob.py')
blast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blast)


def response(text, status=200):
    result = blast.requests.Response()
    result.status_code = status
    result._content = text.encode()
    return result


class BlastTests(unittest.TestCase):
    def test_sequential_jobs_sleep_before_every_check_and_between_submissions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / 'data.fasta'
            fasta.write_text('>one\nACDE\n>bad\nAC*D\n>two\nFGHI\n')
            events = []
            replies = iter([
                'ncbiblast-one', 'RUNNING', 'FINISHED',
                '<types><identifier>outxml</identifier></types>', '<blast/>',
                'ncbiblast-two', 'FINISHED',
                '<types><identifier>xml</identifier></types>', '<blast/>',
            ])
            def request(method, url, **kwargs):
                events.append((method, url.split(blast.BASE_URL)[1]))
                return response(next(replies))
            with patch.multiple(blast, INPUT_PATH=fasta, OUTPUT_DIR=root, EMAIL='test@example.org', BATCH_SIZE=2, POLL_INTERVAL=30), \
                 patch.object(blast.requests, 'Session') as session, \
                 patch.object(blast.time, 'sleep', side_effect=lambda seconds: events.append(('sleep', seconds))), \
                 patch.object(blast.logging, 'basicConfig'), patch.object(blast.logging, 'error') as log:
                session.return_value.__enter__.return_value.request.side_effect = request
                self.assertEqual(blast.main(), 1)  # Invalid record is logged, others finish.
                self.assertIn('bad', log.call_args.args[2])
            self.assertEqual(len(list(root.glob('*.xml'))), 2)
            self.assertEqual(events, [
                ('POST', '/run'), ('sleep', 30), ('GET', '/status/ncbiblast-one'),
                ('sleep', 30), ('GET', '/status/ncbiblast-one'),
                ('GET', '/resulttypes/ncbiblast-one'), ('GET', '/result/ncbiblast-one/outxml'),
                ('sleep', 10), ('sleep', 10), ('POST', '/run'),
                ('sleep', 30), ('GET', '/status/ncbiblast-two'),
                ('GET', '/resulttypes/ncbiblast-two'), ('GET', '/result/ncbiblast-two/xml'),
                ('sleep', 10),
            ])

    def test_wrapped_scheduler_rate_limit_retries_status_with_backoff(self):
        session = Mock()
        throttled = response('<error>rate_limit_exceeded</error>', 400)
        session.request.side_effect = [throttled, throttled, response('FINISHED')]
        with patch.object(blast.time, 'sleep') as sleep, patch.object(blast.logging, 'warning') as log:
            result = blast.request(session, 'GET', '/status/ncbiblast-one')
        self.assertEqual(result.text, 'FINISHED')
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [30, 60])
        self.assertIn('rate_limit_exceeded', log.call_args.args[4])

    def test_other_errors_log_body_without_retrying(self):
        for method, endpoint, status in [('GET', '/status/ncbiblast-one', 400), ('POST', '/run', 429)]:
            with self.subTest(method=method):
                session = Mock()
                session.request.return_value = response('Service error details', status)
                with patch.object(blast.logging, 'error') as log, patch.object(blast.time, 'sleep') as sleep:
                    with self.assertRaises(blast.requests.HTTPError):
                        blast.request(session, method, endpoint)
                self.assertEqual(session.request.call_count, 1)
                self.assertEqual(log.call_args.args[4], 'Service error details')
                sleep.assert_not_called()

    def test_status_retry_limit_and_retry_after(self):
        session = Mock()
        throttled = response('Too many requests', 429)
        throttled.headers['Retry-After'] = '90'
        session.request.return_value = throttled
        with patch.object(blast.time, 'sleep') as sleep, \
             patch.object(blast.logging, 'warning'), patch.object(blast.logging, 'error'):
            with self.assertRaises(blast.requests.HTTPError):
                blast.request(session, 'GET', '/status/ncbiblast-one')
        self.assertEqual(session.request.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [90, 90, 120])

    def test_network_failure_stops_before_submitting_another_protein(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / 'data.fasta'
            fasta.write_text('>one\nACDE\n>two\nFGHI\n')
            with patch.multiple(blast, INPUT_PATH=fasta, OUTPUT_DIR=root, EMAIL='test@example.org'), \
                 patch.object(blast, 'search', side_effect=blast.requests.Timeout('timeout')) as search, \
                 patch.object(blast.logging, 'basicConfig'), patch.object(blast.logging, 'error') as log:
                self.assertEqual(blast.main(), 1)
                self.assertEqual(search.call_count, 1)
                self.assertIn('one', log.call_args.args[2])


if __name__ == '__main__':
    unittest.main()
