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
    assert not any(f['rule'] == 'slopmonster_proof' for f in a['findings'])
    assert a['number_count'] == 1
    assert len(a['summary_notes']) == 1
    assert 'Keep accurate metrics' in a['summary_notes'][0]
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


def test_metrics_only_produce_one_summary_note():
    result = review_writing('- Supported 5 teams, 40 clients, and 12 projects.')
    assert result['findings'] == []
    assert result['number_count'] == 3
    assert len(result['summary_notes']) == 1
    for text in ('Managed 5 teams.', 'Managed 12 cross-functional teams.'):
        review = review_writing(text)
        assert review['findings'] == []
        assert review['number_count'] == 1


def test_repeated_openers_are_scoped_to_each_role():
    result = review_writing('EXPERIENCE\nAnalyst, Acme\n- Developed reports.\n'
                            'Analyst, Beta\n- Developed dashboards.\n'
                            'Analyst, Gamma\n- Developed pipelines.')
    assert not any(f['rule'] == 'repeated_opener' for f in result['findings'])
    result = review_writing('Analyst, Acme\n- Developed reports for\ninternal teams.\n'
                            '- Developed dashboards.\n- Developed pipelines.')
    repeated = [f for f in result['findings'] if f['rule'] == 'repeated_opener']
    assert [f['line'] for f in repeated] == [2, 4, 5]


def test_languages_subheading_preserves_skills_context():
    result = review_writing('SKILLS\nLanguages\nPython, Java, and Scala')
    assert result['findings'] == []
    result = review_writing('SKILLS\nLanguages\nPython, Java, and Scala\nEXPERIENCE\n'
                            '- Led robust, seamless, and transformative programs.')
    assert any(f['rule'] == 'slopmonster_rhythm' for f in result['findings'])


def test_grouped_api_and_plain_rendering_show_one_excerpt():
    from ats_checker.writing_check import writing_review_lines
    text = '- Worked on seamless synergy at the end of the day.'
    result = review_writing(text)
    assert len(result['findings']) >= 3
    assert len(result['grouped_findings']) == 1
    group = result['grouped_findings'][0]
    assert group['excerpt'] == text
    assert len(group['issues']) == len(result['findings'])
    assert all('excerpt' not in issue and 'line' not in issue for issue in group['issues'])
    rendered = '\n'.join(writing_review_lines(result))
    assert rendered.count(text) == 1
    assert 'Extracted line 1' in rendered


def test_unavailable_review_is_never_rendered_as_clean():
    from ats_checker.writing_check import writing_review_lines
    rendered = '\n'.join(writing_review_lines({'status': 'unavailable', 'findings': []}))
    assert 'unavailable' in rendered
    assert 'No local writing issues' not in rendered


def test_cli_local_consent_warns_without_polluting_json(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    path = tmp_path / 'resume.txt'
    path.write_text('- Built Python dashboards.')
    response = subprocess.run([sys.executable, 'cli.py', 'writing-check', '--resume',
                               str(path), '--consent', '--json'], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1])
    assert response.returncode == 0
    assert '--consent has no effect' in response.stderr
    assert json.loads(response.stdout)['detector']['status'] == 'not_run'


def test_realistic_resume_has_only_actionable_contribution_feedback():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'samples/writing_review_resume.txt').read_text()
    result = review_writing(text)
    assert len(result['findings']) == 1
    assert result['findings'][0]['rule'] == 'unclear_contribution'
    assert result['findings'][0]['excerpt'] == '- Worked on dashboards.'


def test_terminal_score_report_groups_excerpt_and_shows_summary():
    from io import StringIO
    from rich.console import Console
    from ats_checker.report import print_report
    from ats_checker.scorer import run_full_check
    excerpt = '- Worked on seamless synergy for 5 teams.'
    result = run_full_check(resume_text=excerpt, jd_text='Python analyst required.',
                            skip_semantic=True, skip_manager=True)
    output = StringIO()
    with patch('rich.console.Console', return_value=Console(file=output, width=2000, color_system=None)):
        print_report(result)
    rendered = output.getvalue()
    assert rendered.count(excerpt) == 1
    assert 'Extracted line 1' in rendered
    assert 'Keep accurate metrics' in rendered


def test_browser_rendering_groups_escapes_and_handles_unavailable():
    import json
    import shutil
    import subprocess
    from server import app
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed to execute the browser renderer')
    with app.test_client() as client:
        page = client.get('/').get_data(as_text=True)
    escape_fn = page[page.index('const esc ='):page.index('function showTab')]
    render_fn = page[page.index('function renderWritingReview'):page.index('async function runWritingCheck')]
    result = review_writing('- Worked on seamless synergy for 5 teams <img src=x onerror=alert(1)>.')
    script = escape_fn + render_fn + '\nprocess.stdout.write(JSON.stringify([' + \
        'renderWritingReview(' + json.dumps(result) + '),' + \
        'renderWritingReview({status:"unavailable",findings:[],message:"Failed <script>"})' + ']));'
    rendered, unavailable = json.loads(subprocess.check_output([node, '-e', script], text=True))
    assert rendered.count('&lt;img') == 1
    assert '<img' not in rendered
    assert 'Extracted line 1' in rendered
    assert 'Keep accurate metrics' in rendered
    assert 'Failed &lt;script&gt;' in unavailable
    assert 'No issues found' not in unavailable


def test_cli_jd_exemptions_and_grouping(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    path = tmp_path / 'resume.txt'
    jd = tmp_path / 'jd.txt'
    path.write_text('- Built seamless systems.')
    jd.write_text('Build seamless systems using Python.')
    root = Path(__file__).resolve().parents[1]
    response = subprocess.run([sys.executable, 'cli.py', 'writing-check', '--resume',
                               str(path), '--jd', str(jd), '--json'], capture_output=True, text=True, cwd=root)
    assert response.returncode == 0
    assert json.loads(response.stdout)['findings'] == []
    path.write_text('- Worked on seamless synergy for 5 teams.')
    response = subprocess.run([sys.executable, 'cli.py', 'writing-check', '--resume',
                               str(path)], capture_output=True, text=True, cwd=root)
    assert response.returncode == 0
    assert response.stdout.count(path.read_text()) == 1
    assert 'Extracted line 1' in response.stdout
    assert 'Keep accurate metrics' in response.stdout


def test_buzzword_tricolon_flagged_but_tools_list_is_not():
    for text in ('Innovative, strategic, and visionary leader.',
                 'Strategic, innovative, and dynamic leader.',
                 'Strategic, proactive, and capable leader.'):
        assert 'slopmonster_rhythm' in {f['rule'] for f in review_writing(text)['findings']}
    for text in ('- Built dashboards using Python, Excel, and Tableau.',
                 '- Built proactive, strategic, and Python systems.'):
        assert 'slopmonster_rhythm' not in {f['rule'] for f in review_writing(text)['findings']}


def test_metric_count_excludes_phone_years_and_date_ranges():
    text = '+91 98765 43210\nJan 2019 - Mar 2021\nBuilt 3 dashboards'
    assert review_writing(text)['number_count'] == 1
    for contact in ('+1 (555) 123-4567', 'Phone: 9876543210', '9876543210'):
        for dates in ('01/2019 - 03/2021', '2019-01 - 2021-03', '15/01/2019 - 03/20/2021'):
            result = review_writing(f'{contact}\n{dates}\nBuilt 3 dashboards and saved 12.5%.')
            assert result['number_count'] == 2
    result = review_writing('Processed 1000000000 records, 12,000 events, and 40 clients.')
    assert result['number_count'] == 3
    assert review_writing('2019 - 2021 - 3 dashboards delivered.')['number_count'] == 1


def test_capitalized_bullet_wrap_preserves_role_and_first_line():
    text = '- Worked on reporting for the\nPython migration\n- Worked on X\n- Worked on Y'
    result = review_writing(text)
    repeated = [f for f in result['findings'] if f['rule'] == 'repeated_opener']
    assert [f['line'] for f in repeated] == [1, 3, 4]
    assert repeated[0]['excerpt'] == '- Worked on reporting for the Python migration'
    for header in ('Analyst, Acme', 'Analyst | Acme', 'Analyst / Acme', 'Acme 2020 - 2021',
                   '\nAcme Corporation'):
        text = f'- Worked on reports\n{header}\n- Worked on X\n- Worked on Y'
        assert not any(f['rule'] == 'repeated_opener' for f in review_writing(text)['findings'])
    text = 'Analyst, Acme | 2019 - 2020\n- Developed reports\n'
    text += 'Analyst, Beta | 2020 - 2021\n- Developed dashboards\n'
    text += 'Analyst, Gamma | 2021 - Present\n- Developed pipelines'
    assert not any(f['rule'] == 'repeated_opener' for f in review_writing(text)['findings'])


def test_undated_company_headers_do_not_merge_with_bullets():
    from ats_checker.writing_check import _review_lines
    text = 'ACME\n- Developed A\nBETA Corp\n- Developed B\nGamma Inc\n- Developed C'
    assert review_writing(text)['findings'] == []
    assert [line for _, line in _review_lines(text)] == text.splitlines()
    wrapped = '- Worked on reporting for the\nPython migration\n- Worked on X\n- Worked on Y'
    result = review_writing(wrapped)
    assert sum(f['rule'] == 'repeated_opener' for f in result['findings']) == 3
    assert result['grouped_findings'][0]['excerpt'] == '- Worked on reporting for the Python migration'


def test_abbreviated_metrics_count_without_contacts_or_dates():
    assert review_writing('- Cut cost 40% and saved $2M across 12 teams')['number_count'] == 3
    assert review_writing('- Reached 5K users, grew 3x, saved 50L and generated 2mn, 4cr and 1.5B.')['number_count'] == 6
    assert review_writing('+91 98765 43210\nJan 2019 - Mar 2021\nBuilt 3 dashboards')['number_count'] == 1
    assert review_writing('Used Python3, m365, S3, 2FA and 3rd-party integrations.')['number_count'] == 0
