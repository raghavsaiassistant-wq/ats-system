from unittest.mock import Mock, patch

from ats_checker.writing_check import check_writing, review_writing


def test_local_review_never_calls_detector():
    with patch('ats_checker.writing_check.requests.post') as request:
        result = review_writing('Results-driven analyst\n- Worked on reports\n- Worked on dashboards\n- Worked on data')
    request.assert_not_called()
    assert result['detector']['status'] == 'not_run'
    assert {f['rule'] for f in result['findings']} == {'generic_phrase', 'unclear_contribution', 'repeated_opener'}
    assert all(f['excerpt'] for f in result['findings'])


def test_external_requires_actual_boolean_consent():
    with patch('ats_checker.writing_check.requests.post') as request:
        for consent in (False, 'true', 1):
            assert check_writing('word ' * 160, 'gptzero', consent)['detector']['status'] == 'not_run'
    request.assert_not_called()


def test_invalid_external_result_is_unavailable(monkeypatch):
    monkeypatch.setenv('ATS_GPTZERO_API_KEY', 'test-key')
    response = Mock(status_code=200)
    response.json.return_value = {'documents': [{'document_classification': 'HUMAN_ONLY', 'class_probabilities': {'human': True, 'mixed': 0, 'ai': 0}}]}
    with patch('ats_checker.writing_check.requests.post', return_value=response):
        result = check_writing('word ' * 160, 'gptzero', True)
    assert result['detector']['status'] == 'unavailable'
    assert 'vendor_label' not in result['detector']


def test_route_local_and_invalid_consent():
    from server import app
    with app.test_client() as client, patch('ats_checker.writing_check.requests.post') as request:
        result = client.post('/writing-check', json={'resume_text': 'Highly motivated analyst'})
        assert result.status_code == 200
        assert result.json['detector']['status'] == 'not_run'
        assert result.json['findings'][0]['rule'] == 'generic_phrase'
        assert client.post('/writing-check', json={'resume_text': 'text', 'consent': 'yes'}).status_code == 400
    request.assert_not_called()


def test_slopmonster_filler_rewrite_preserves_numbers_and_tools():
    text = '- At the end of the day, analyzed 120 reports in Power BI.'
    findings = review_writing(text)['findings']
    f = next(f for f in findings if f['rule'] == 'slopmonster_phrases')
    assert f['suggested_rewrite'] == '- Analyzed 120 reports in Power BI.'
    assert f['requires_confirmation'] is True
    assert f['excerpt'] == text


def test_slopmonster_spacing_normalized_and_metrics_are_advisory():
    a = review_writing('A game-changing journey for 12 teams.')
    b = review_writing('A  game-changing   journey for 12 teams.')
    assert [f['rule'] for f in a['findings']] == [f['rule'] for f in b['findings']]
    proof = next(f for f in a['findings'] if f['rule'] == 'slopmonster_proof')
    assert proof['suggested_rewrite'] is None
    assert 'Keep it if accurate' in proof['suggestion']
    assert a['detector']['status'] == 'not_run'
    assert 'score' not in a


def test_slopmonster_preserves_technical_terms_and_skills_lists():
    result = review_writing('SKILLS\nPython, Excel, and Tableau\nEXPERIENCE\n- Built robust data transformation pipelines.')
    assert not any(f['rule'] in ('slopmonster_rhythm', 'slopmonster_vocab') for f in result['findings'])


def test_route_returns_recommendation_and_escaped_ui_fields():
    from server import app
    with app.test_client() as client:
        response = client.post('/writing-check', json={'resume_text': 'At the end of the day, analyzed reports.'})
        assert response.status_code == 200
        f = next(f for f in response.json['findings'] if f['rule'] == 'slopmonster_phrases')
        assert f['suggested_rewrite'] == 'Analyzed reports.'
        page = client.get('/').get_data(as_text=True)
        assert 'esc(f.suggested_rewrite)' in page
        assert 'esc(f.suggestion' in page
