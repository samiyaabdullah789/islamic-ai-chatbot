
"""Conservative six-field SeekersGuidance JSON cleaner. Run: python scripts/clean_json.py"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw_json'
OUT = ROOT / 'data' / 'cleaned_json'
REVIEW = OUT / 'review'

NAME_TITLE = r'(?:\[?(?:shaykh|sheikh|ustadha?|mufti|mawlana|imam|sidi)\]?\s*)*'
SALAM = re.compile(r'(?i)^(?:wa\s*)?(?:as[- ]?)?sal[aā]+m(?:u|un|oon|aam|am)?(?:\s+ala[iy]kum)?(?:\s+wa\s+rahmat\w*)?(?:\s+wa\s+barakat\w*)?[\s,!.:;-]*$|^(?:wa\s+ala[iy]kum\s+(?:as[- ]?)?sal[aā]+m)[\s,!.:;-]*$', re.I)
SALAM_PREFIX = re.compile(r'(?i)^\s*(?:wa\s+ala[iy]kum\s+(?:as[- ]?)?sal[aā]+m|(?:as[- ]?)?sal[aā]+mu?\s+ala[iy]kum)(?:\s+wa\s+rahmat\w*)?(?:\s+wa\s+barakat\w*)?[\s,!.:;-]*')
WASSALAM = re.compile(r'(?i)^\s*(?:wa\s*ss?al+a+m|wass?al+a+m|wa\s+sal[aā]+m)(?:\s+ala[iy]kum)?[\s,!.:;-]*(.*)$')
ALLAH_KNOWS = re.compile(r'(?i)^(?:and\s+)?allah(?:\s+most\s+high)?\s+knows\s+best(?:\s+and\s+he\s+alone\s+gives\s+success)?[\s.!]*$')
BIO = re.compile(r'(?i)\b(?:received ijazah|was born|born and raised|grew up|graduated from|studied at|inspires her students|received his education|received her education|holds a degree|is a graduate of|studied islamic sciences|is a teacher at|is an instructor at)\b')
POLITE = re.compile(r'(?i)^(?:i\s+(?:pray|hope)\s+(?:that\s+)?(?:you|this)\b.{0,170}|(?:thank you|jazak\w*\s+khair)\s+for\s+(?:your|the)\s+question\b.{0,80}|dear\s+(?:brother|sister|questioner)[\s,!.]*|i\s+hope\s+this\s+helps[\s.!]*)$')
APPROVAL = re.compile(r'(?i)^\s*(?:checked\s*(?:&|and)\s*approved|reviewed|checked)\s+by\s+.{1,130}$')
ANSWER_MARKER = re.compile(r'(?i)\banswer\s*:\s*')
ARABIC = re.compile(r'[\u0600-\u06ff]')


def tidy(v):
    if not isinstance(v, str):
        return ''
    v = v.replace('\r\n', '\n').replace('\r', '\n').replace('\u00a0', ' ')
    v = re.sub(r'[ \t]+', ' ', v)
    return re.sub(r'\n{3,}', '\n\n', v).strip()


def as_list(v):
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            v = parsed if isinstance(parsed, list) else [v]
        except (ValueError, TypeError):
            v = [v] if v.strip() else []
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    seen, result = set(), []
    for x in v:
        x = tidy(x if isinstance(x, str) else str(x))
        if x and x.casefold() not in seen:
            seen.add(x.casefold())
            result.append(x)
    return result


def scholar_names(rec):
    result = []
    for k in ('scholar', 'answered_by'):
        s = tidy(rec.get(k))
        if not s or len(s) > 140:
            continue
        result.append(s)
        plain = re.sub(r'(?i)^' + NAME_TITLE, '', s).strip()
        plain = re.sub(r'\s*\(.*$', '', plain).strip()
        if plain:
            result.append(plain)
    return sorted(set(result), key=len, reverse=True)


def is_scholar_signature(line, rec):
    line = tidy(line).strip(' ,.;:-')
    if not line or len(line) > 155 or ARABIC.search(line):
        return False
    names = scholar_names(rec)
    for name in names:
        if line.casefold() == name.casefold():
            return True
        plain = re.sub(r'(?i)^' + NAME_TITLE, '', line).strip()
        if plain.casefold() == name.casefold():
            return True
        if re.fullmatch(
            re.escape(name) + r'\s+(?:\w+\s+\d{1,2},?\s+\d{4})(?:/.*)?',
            line,
            re.I
        ):
            return True
    return False


def signoff_line(line, rec):
    m = WASSALAM.match(line)
    if not m:
        return False
    rest = tidy(m.group(1)).strip(' ,.;:-')
    return not rest or is_scholar_signature(rest, rec)


def is_bio(paragraph, rec):
    p = tidy(paragraph)
    if not BIO.search(p) or ARABIC.search(p):
        return False
    start = p[:180].casefold()
    return any(
        start.startswith(name.casefold()) or
        bool(re.match(r'(?i)^' + NAME_TITLE + re.escape(name) + r'\b', p))
        for name in scholar_names(rec) if len(name) > 5
    )


def clean_answer(raw, rec):
    text = tidy(raw)
    flags = []
    if not text:
        return '', flags

    paragraphs = [
        p.strip()
        for p in re.split(r'\n\s*\n', text)
        if p.strip()
    ]

    # Remove opening conversational text only.
    for _ in range(8):
        if not paragraphs:
            break

        first = paragraphs[0]
        lines = first.split('\n')
        leading = lines[0].strip()

        new = SALAM_PREFIX.sub('', leading, count=1)

        if new != leading and new.strip():
            lines[0] = new.strip()
            paragraphs[0] = '\n'.join(lines)
            flags.append('opening_greeting_removed')
            continue

        if SALAM.fullmatch(leading) or POLITE.fullmatch(leading):
            lines.pop(0)
            if lines:
                paragraphs[0] = '\n'.join(lines)
            else:
                paragraphs.pop(0)
            flags.append('opening_boilerplate_removed')
            continue

        break

    kept = []

    for para in paragraphs:
        if is_bio(para, rec):
            flags.append('scholar_biography_removed')
            continue

        lines = []

        for line in para.split('\n'):
            l = line.strip()

            if re.fullmatch(
                r'(?i)(?:cover photo|photo|image)\s+by\s+.{2,120}',
                l
            ):
                flags.append('photo_credit_removed')
                continue

            if APPROVAL.fullmatch(l):
                flags.append('approval_removed')
                continue

            if signoff_line(l, rec):
                flags.append('signoff_removed')
                continue

            # Remove combined closing and signature.
            if re.match(
                r'(?i)^(?:and\s+)?allah\s+knows\s+best\.?\s+',
                l
            ):
                rest = re.sub(
                    r'(?i)^(?:and\s+)?allah\s+knows\s+best\.?\s+',
                    '',
                    l
                )

                if signoff_line(rest, rec) or is_scholar_signature(rest, rec):
                    flags.append('closing_signature_removed')
                    continue

            lines.append(l)

        if lines:
            kept.append('\n'.join(lines))

    # Remove ending boilerplate.
    for _ in range(12):
        if not kept:
            break

        lines = kept[-1].split('\n')
        tail = lines[-1].strip()
        reason = None

        if re.fullmatch(
            r'(?i)(?:cover photo|photo|image)\s+by\s+.{2,120}',
            tail
        ):
            reason = 'photo_credit_removed'

        elif re.fullmatch(
            r"(?i)(?:ma[’']?salam|with salaams)[,!. ]*",
            tail
        ):
            reason = 'signoff_removed'

        elif is_scholar_signature(tail, rec):
            reason = 'scholar_signature_removed'

        elif signoff_line(tail, rec):
            reason = 'signoff_removed'

        elif ALLAH_KNOWS.fullmatch(tail):
            reason = 'closing_removed'

        elif re.fullmatch(
            r'(?i)i\s+hope\s+(?:this|that\s+this)\s+helps[\s.!]*',
            tail
        ):
            reason = 'closing_removed'

        if not reason:
            break

        flags.append(reason)
        lines.pop()

        if lines:
            kept[-1] = '\n'.join(lines)
        else:
            kept.pop()

    # Remove clear trailing promotions.
    while kept and re.match(
        r'(?i)^(?:related answers?|cover photo|love, marriage and relationships in islam:)\b',
        kept[-1]
    ):
        block = kept[-1]

        if block.lower().startswith('related answer'):
            flags.append('related_links_review')
            break

        kept.pop()
        flags.append('trailing_promotion_removed')

    answer = tidy('\n\n'.join(kept))

    # Uncertain content is flagged rather than deleted.
    if WASSALAM.search(answer) or re.search(
        r'(?im)^\s*(?:and\s+)?allah\s+knows\s+best\b',
        answer
    ):
        flags.append('remaining_signoff_or_closing_review')

    if BIO.search(answer):
        flags.append('possible_biography_review')

    return answer, list(dict.fromkeys(flags))


def clean_record(rec):
    if not isinstance(rec, dict):
        return None, ['invalid_record']

    issues = []

    question = tidy(rec.get('question'))
    answer = tidy(rec.get('answer'))

    # Separate an embedded duplicate answer where safe.
    m = ANSWER_MARKER.search(question)

    if m:
        before = tidy(question[:m.start()])
        embedded = tidy(question[m.end():])

        if before and embedded:
            if not answer:
                question = before
                answer = embedded
                issues.append('answer_recovered_from_question')

            elif embedded in answer or answer in embedded:
                question = before
                issues.append('duplicate_answer_removed_from_question')

            else:
                issues.append('embedded_answer_needs_review')

    answer, flags = clean_answer(answer, rec)
    issues.extend(flags)

    scholar = (
        tidy(rec.get('scholar'))
        or tidy(rec.get('answered_by'))
    )

    if len(scholar) > 180 or '\n' in scholar:
        issues.append('scholar_metadata_review')
        scholar = ''

    output = {
        'question': question,
        'answer': answer,
        'scholar': scholar,
        'madhhab': tidy(rec.get('madhhab')),
        'categories': as_list(rec.get('categories')),
        'tags': as_list(rec.get('tags')),
    }

    if not question:
        issues.append('missing_question')

    if not answer:
        issues.append('missing_answer')

    return output, list(dict.fromkeys(issues))


def process(path):
    try:
        obj = json.loads(
            path.read_text(encoding='utf-8-sig')
        )

        records = (
            obj if isinstance(obj, list)
            else obj['records']
        )

        if not isinstance(records, list):
            raise ValueError('records is not a list')

    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR {path.name}: {exc}')
        return

    cleaned = []
    flagged = []
    counts = Counter()

    for i, record in enumerate(records, 1):
        item, issues = clean_record(record)

        if item is not None:
            cleaned.append(item)

        if issues:
            counts.update(issues)
            flagged.append({
                'record_number': i,
                'issues': issues,
                'original': record,
                'cleaned': item,
            })

    output_path = OUT / f'{path.stem}_cleaned.json'
    review_path = REVIEW / f'{path.stem}_review.json'

    output_path.write_text(
        json.dumps(
            cleaned,
            ensure_ascii=False,
            indent=2
        ),
        encoding='utf-8'
    )

    review_path.write_text(
        json.dumps(
            {
                'source_file': path.name,
                'original_records': len(records),
                'cleaned_records': len(cleaned),
                'flagged_records': len(flagged),
                'issue_counts': dict(counts),
                'records': flagged,
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding='utf-8'
    )

    print(
        f'{path.name}: '
        f'original={len(records)}, '
        f'cleaned={len(cleaned)}, '
        f'flagged={len(flagged)}'
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    REVIEW.mkdir(
        parents=True,
        exist_ok=True
    )

    paths = sorted(RAW.glob('*.json'))

    if not paths:
        print(f'No JSON files found in {RAW}')
        return

    for path in paths:
        process(path)

    print('Done. Original files were not modified.')


if __name__ == '__main__':
    main()
