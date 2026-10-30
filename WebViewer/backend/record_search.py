"""Search raw loader records before truncation and matching-record pagination."""
import heapq
import json
import math
from itertools import islice

OPERATIONS = {'contains', 'not_contains', 'equals', 'starts', 'ends', 'gte', 'lte', 'empty', 'not_empty'}


def parse_search(raw):
    options = json.loads(raw) if raw else {}
    if not isinstance(options, dict):
        raise ValueError('Search must be an object')
    if not isinstance(options.get('query', ''), str) or not isinstance(options.get('sort', ''), str):
        raise ValueError('Search query and sort must be strings')
    for key in ('case_sensitive', 'descending'):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    filters = options.get('filters', [])
    if not isinstance(filters, list) or len(filters) > 32:
        raise ValueError('Search supports up to 32 field filters')
    for item in filters:
        if not isinstance(item, dict) or not isinstance(item.get('column'), str) or item.get('operation') not in OPERATIONS or not isinstance(item.get('value', ''), str):
            raise ValueError('Invalid field filter')
        if item['operation'] in {'gte', 'lte'}:
            try:
                if not math.isfinite(float(item.get('value', ''))):
                    raise ValueError()
            except ValueError:
                raise ValueError('Numeric comparisons require a finite number') from None
    return options


def text(value):
    if value is None:
        return ''
    if isinstance(value, dict):
        return ' '.join(f'{key} {text(item)}' for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ' '.join(text(item) for item in value)
    return str(value)


def as_row(record):
    return record if isinstance(record, dict) else {'value': record}


def matches(record, options):
    normalize = (lambda value: text(value)) if options.get('case_sensitive') else (lambda value: text(value).casefold())
    row = as_row(record)
    query = normalize(options.get('query', ''))
    if query and not any(query in normalize(value) for value in row.values()):
        return False
    for item in options.get('filters', []):
        source = normalize(row.get(item['column']))
        target = normalize(item.get('value', ''))
        operation = item['operation']
        if operation in {'gte', 'lte'}:
            try:
                number, bound = float(source), float(target)
                matched = math.isfinite(number) and math.isfinite(bound) and (number >= bound if operation == 'gte' else number <= bound)
            except ValueError:
                matched = False
        else:
            matched = {
                'contains': lambda: target in source,
                'not_contains': lambda: target not in source,
                'equals': lambda: source == target,
                'starts': lambda: source.startswith(target),
                'ends': lambda: source.endswith(target),
                'empty': lambda: not source.strip(),
                'not_empty': lambda: bool(source.strip()),
            }[operation]()
        if not matched:
            return False
    return True


def record_page(iterator, offset, limit, options):
    columns = []

    def matching():
        for record in iterator:
            for column in as_row(record):
                if column not in columns and len(columns) < 32:
                    columns.append(column)
            if matches(record, options):
                yield record

    try:
        records = matching()
        if options.get('sort'):
            def key(record):
                value = text(as_row(record).get(options['sort']))
                try:
                    number = float(value)
                    if math.isfinite(number):
                        return (0, number)
                except ValueError:
                    pass
                return (1, value if options.get('case_sensitive') else value.casefold())
            select = heapq.nlargest if options.get('descending') else heapq.nsmallest
            page = select(offset + limit + 1, records, key=key)[offset:]
        else:
            page = list(islice(records, offset, offset + limit + 1))
        return page[:limit], len(page) > limit, columns
    finally:
        close = getattr(iterator, 'close', None)
        if close:
            close()
