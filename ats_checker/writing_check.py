"""Explainable local writing review and an opt-in experimental detector.

Local observations are not evidence of AI authorship. External responses keep
the provider's semantics and never feed eligibility or scoring calculations.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from datetime import datetime, timezone

import requests

LIMITATION = ('Writing patterns cannot prove who wrote a resume. No signal does not '
              'prove human authorship. This check does not predict selection.')
MIN_DETECTOR_WORDS = 150  # conservative app policy, not a vendor accuracy guarantee
MAX_DETECTOR_CHARACTERS = 50000  # bounded app request size


def review_writing(text: str) -> dict:
    findings = []
    bullets = []
    for line_number, line in enumerate(text.splitlines(), 1):
        excerpt = line.strip()
        if not excerpt:
            continue
        if re.search(r'\b(results[- ]driven|dynamic professional|highly motivated|'
                     r'proven track record|synerg(?:y|ies))\b', excerpt, re.I):
            findings.append({'line': line_number, 'excerpt': excerpt, 'rule': 'generic_phrase',
                             'explanation': 'Replace broad self-description with specific experience you can support.'})
        if re.match(r'^\s*[-*•]\s+', line):
            bullet = re.sub(r'^\s*[-*•]\s+', '', line).strip()
            bullets.append((line_number, excerpt, bullet))
            if re.match(r'(responsible for|helped with|worked on)\b', bullet, re.I):
                findings.append({'line': line_number, 'excerpt': excerpt, 'rule': 'unclear_contribution',
                                 'explanation': 'Explain your contribution and what changed; add only facts you can defend.'})
    openers = {}
    for n, excerpt, bullet in bullets:
        words = re.findall(r'[A-Za-z]+', bullet.lower())
        if words:
            openers.setdefault(words[0], []).append((n, excerpt))
    for entries in openers.values():
        if len(entries) >= 3:
            for n, excerpt in entries:
                findings.append({'line': n, 'excerpt': excerpt, 'rule': 'repeated_opener',
                                 'explanation': 'Several bullets share this opening; consider wording that describes each actual action.'})
    return {'kind': 'local_writing_review', 'status': 'completed', 'findings': findings,
            'word_count': len(text.split()), 'limitation': LIMITATION,
            'detector': {'status': 'not_run', 'provider': None,
                         'message': 'No external detector was run. Local feedback is not AI detection.'}}


def check_writing(text: str, provider: str = 'local', consent: bool = False) -> dict:
    result = review_writing(text)
    if provider == 'local':
        return result
    if provider != 'gptzero':
        raise ValueError('Supported writing providers: local, gptzero')
    reading = {'status': 'not_run', 'provider': 'GPTZero', 'experimental': True,
               'limitation': LIMITATION}
    result['detector'] = reading
    if consent is not True:
        reading['message'] = 'Explicit consent is required to send this resume text to GPTZero.'
        return result
    words = len(text.split())
    if words < MIN_DETECTOR_WORDS:
        reading.update(status='insufficient_text', message=f'At least {MIN_DETECTOR_WORDS} words are required by this app policy; this does not guarantee suitable prose.')
        return result
    if len(text) > MAX_DETECTOR_CHARACTERS:
        reading.update(status='unavailable', message='Text exceeds the app request limit; nothing was sent or truncated.')
        return result
    key = os.environ.get('ATS_GPTZERO_API_KEY', '').strip()
    if not key:
        reading.update(status='unavailable', message='Configure the GPTZero API key on the server. Nothing was sent.')
        return result
    payload = {'document': text}
    requested_version = os.environ.get('ATS_GPTZERO_VERSION', '').strip()
    if requested_version:
        payload['version'] = requested_version
    try:
        response = requests.post('https://api.gptzero.me/v2/predict/text',
                                 headers={'x-api-key': key, 'Accept': 'application/json'},
                                 json=payload, timeout=(5, 30), allow_redirects=False)
        if response.status_code != 200:
            reading.update(status='unavailable', message=f'Detector request failed (HTTP {response.status_code}); no authorship assessment was made.')
            return result
        raw = response.json()
        if not isinstance(raw, dict):
            raise ValueError('invalid response')
        documents = raw.get('documents')
        doc = documents[0] if isinstance(documents, list) and len(documents) == 1 else raw
        if not isinstance(doc, dict):
            raise ValueError('invalid document')
        label = doc.get('document_classification')
        if label not in ('HUMAN_ONLY', 'MIXED', 'AI_ONLY'):
            raise ValueError('missing or unknown classification')
        probs = doc.get('class_probabilities')
        if not isinstance(probs, dict) or set(probs) != {'human', 'mixed', 'ai'}:
            raise ValueError('missing class probabilities')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
            raise ValueError('invalid class probabilities')
        if not math.isclose(sum(probs.values()), 1, abs_tol=.02):
            raise ValueError('invalid probability total')
        messages = {'HUMAN_ONLY': 'No strong AI signal detected by this provider.',
                    'MIXED': 'Provider reports mixed human/AI patterns; authorship remains uncertain.',
                    'AI_ONLY': 'Provider detected AI-like patterns; this is not proof of authorship.'}
        version = doc.get('model_version') or raw.get('model_version')
        reading.update(status='completed', vendor_label=label, message=messages[label],
                       class_probabilities=probs,
                       score_semantics='Provider class confidence, not a percentage of words generated or a validated resume authorship probability.',
                       returned_model_version=version if isinstance(version, str) else None,
                       requested_model_version=requested_version or None,
                       checked_at=datetime.now(timezone.utc).isoformat(),
                       text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                       words_sent=words)
    except (requests.RequestException, ValueError, TypeError, KeyError):
        # Never expose raw vendor bodies or exceptions containing credentials/text.
        reading.update(status='unavailable', message='Detector unavailable or its response could not be validated; no authorship assessment was made.')
    return result
