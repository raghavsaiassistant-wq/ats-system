from unittest.mock import Mock, patch

import pytest

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


def test_skills_section_ends_at_common_heading_variants():
    for heading in ('WORK EXPERIENCE', 'Employment History', 'Projects:'):
        result = review_writing(f'SKILLS\nPython, Excel\n{heading}\n- Led robust, seamless, and transformative programs.')
        assert 'slopmonster_rhythm' in {f['rule'] for f in result['findings']}, heading


def test_synergy_reported_once():
    rules = [f['rule'] for f in review_writing('Synergy-focused, results-driven lead')['findings']]
    assert rules == ['generic_phrase']


@pytest.mark.parametrize('text', [
    '- Streamlined reporting.', '- Leveraged SQL.', '- Fostered innovation.',
    '- Showcased and empowered process optimization.',
    '- Built dashboards using Python, Excel, and Tableau.',
    '- Built APIs using Postgres, NodeJS, and Kubernetes.',
    'Technical Skills: Python, Excel, and Tableau',
    'CORE COMPETENCIES\nPython, Excel, and Tableau',
    'Tech Stack\nPython, Excel, and Tableau',
    'Tools & Technologies\nPython, Excel, and Tableau',
    '- Built real-time, event-driven, end-to-end, cross-functional systems.',
])
def test_resume_terminology_is_not_weak_writing(text):
    assert review_writing(text)['findings'] == []


def test_jd_vocabulary_is_preserved_in_standalone_and_score_reviews():
    from ats_checker.scorer import run_full_check
    text = '- Built seamless systems.'
    jd = 'Requirements: build seamless systems using Python.'
    assert any(f['rule'] == 'slopmonster_vocab' for f in review_writing(text)['findings'])
    assert review_writing(text, jd)['findings'] == []
    report = run_full_check(resume_text=text, jd_text=jd, skip_semantic=True, skip_manager=True)
    assert report.writing_review['findings'] == []


@pytest.mark.parametrize('marker', ['-', '*', '•', '●', '▪', '◦', '–', '➢', '►', 'o'])
def test_exported_bullet_markers(marker):
    result = review_writing('\n'.join(f'{marker} Worked on {item}' for item in ('reports', 'dashboards', 'data')))
    assert sum(f['rule'] == 'unclear_contribution' for f in result['findings']) == 3
    assert sum(f['rule'] == 'repeated_opener' for f in result['findings']) == 3


@pytest.mark.parametrize('explicit_numbering', [False, True])
def test_docx_list_paragraphs_keep_bullets(tmp_path, explicit_numbering):
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from ats_checker.parsing import analyze
    document = docx.Document()
    for item in ('reports', 'dashboards', 'data'):
        para = document.add_paragraph(f'Worked on {item}', style='Normal' if explicit_numbering else 'List Bullet')
        if explicit_numbering:
            num_pr = para._p.get_or_add_pPr().get_or_add_numPr()
            num_id = OxmlElement('w:numId')
            num_id.set(qn('w:val'), '1')
            num_pr.append(num_id)
    path = tmp_path / 'resume.docx'
    document.save(path)
    parsed = analyze(path=str(path))
    assert parsed.text.startswith('- Worked on reports')
    result = review_writing(parsed.text)
    assert sum(f['rule'] == 'unclear_contribution' for f in result['findings']) == 3
    assert sum(f['rule'] == 'repeated_opener' for f in result['findings']) == 3


def test_optional_review_failure_preserves_scores():
    from ats_checker.scorer import run_full_check
    kwargs = dict(resume_text='- Built Python dashboards.', jd_text='Python analyst required.',
                  skip_semantic=True, skip_manager=True)
    baseline = run_full_check(**kwargs)
    with patch('ats_checker.scorer.review_writing', side_effect=RuntimeError('private resume text')):
        result = run_full_check(**kwargs)
    assert result.ats_score == baseline.ats_score
    assert result.recruiter_score == baseline.recruiter_score
    assert result.writing_review['status'] == 'unavailable'
    assert 'private resume text' not in str(result.to_dict())
    assert 'unavailable' in result.notes[-1]


def test_resume_cliches_are_identified_individually():
    result = review_writing('Detail-oriented team player, self-starter, hard-working and passionate analyst.')
    finding = next(f for f in result['findings'] if f['rule'] == 'generic_phrase')
    assert {m.lower() for m in finding['matches']} == {
        'detail-oriented', 'team player', 'self-starter', 'hard-working', 'passionate'}


@pytest.mark.parametrize('opener', ['Assisted with', 'Was responsible for', 'Responsible for',
                                    'Involved in', 'Participated in', 'Duties included'])
def test_weak_contribution_openers(opener):
    result = review_writing(f'▪ {opener} reporting and data analysis.')
    assert any(f['rule'] == 'unclear_contribution' for f in result['findings'])


@pytest.mark.parametrize('text', ['Mapped customer journey.', 'Mapped user journey.',
                                  'Built curated datasets.', 'Completed a deep dive into churn data.'])
def test_domain_vocabulary_is_preserved(text):
    assert review_writing(text)['findings'] == []


@pytest.mark.parametrize('marker', ['', '- ', '● ', '▪ '])
def test_wrapped_filler_is_found_and_rewrite_keeps_facts(marker):
    result = review_writing(f'EXPERIENCE\n{marker}At the end of\nthe day, analyzed 120 reports in Power BI.')
    finding = next(f for f in result['findings'] if f['rule'] == 'slopmonster_phrases')
    assert finding['line'] == 2
    assert finding['suggested_rewrite'] == f'{marker}Analyzed 120 reports in Power BI.'


def test_wrapped_lines_do_not_cross_headings_or_blank_lines():
    for boundary in ('\n\n', '\nEXPERIENCE\n', '\nSKILLS\n'):
        result = review_writing('At the end of' + boundary + 'the day')
        assert not any(f['rule'] == 'slopmonster_phrases' for f in result['findings'])


def test_external_route_rejects_file_paths_before_reading_or_sending(tmp_path):
    from server import app
    path = tmp_path / 'resume.txt'
    path.write_text('word ' * 160)
    with app.test_client() as client, patch('ats_checker.writing_check.requests.post') as request, \
            patch('server.parsing.analyze') as parse:
        for consent in (False, True):
            response = client.post('/writing-check', json={
                'resume_path': str(path), 'resume_text': 'word ' * 160,
                'provider': 'gptzero', 'consent': consent})
            assert response.status_code == 400
            assert 'file-path' in response.json['error']
    request.assert_not_called()
    parse.assert_not_called()
    with app.test_client() as client:
        assert client.post('/writing-check', json={'resume_path': str(path)}).status_code == 200


def test_cli_missing_resume_has_clean_error_and_exit_two(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    response = subprocess.run([sys.executable, 'cli.py', 'writing-check', '--resume',
                               str(tmp_path / 'missing.pdf')], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1])
    assert response.returncode == 2
    assert 'Resume file not found' in response.stderr
    assert 'Traceback' not in response.stderr
