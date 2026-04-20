from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any

import yaml

try:
    import certifi
except ImportError:  # pragma: no cover - 兼容未安装 certifi 的环境
    certifi = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/123.0.0.0 Safari/537.36'
)


@dataclass(frozen=True)
class HelpPageEntry:
    help_category_id: str
    help_category_name: str
    item_id: str
    name: str
    order: int
    detail_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fetch help pages referenced by assets/gakumasu-diff/HelpContent.yaml.')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=Path('docs/help_content_pages'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=20.0)
    parser.add_argument('--limit', type=int, default=0, help='Only fetch the first N matched pages. 0 means all.')
    parser.add_argument(
        '--category',
        action='append',
        default=[],
        help='Filter by help category id. Can be passed multiple times.',
    )
    parser.add_argument(
        '--match',
        default='',
        help='Only fetch rows whose id/name/url matches this case-insensitive substring.',
    )
    parser.add_argument(
        '--refresh',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Refetch pages even if cached extracted text exists.',
    )
    return parser.parse_args()


def _load_yaml_rows(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'Expected list in {path}')
    return payload


def _load_entries(root: Path) -> list[HelpPageEntry]:
    diff_root = root / 'assets' / 'gakumasu-diff'
    categories = _load_yaml_rows(diff_root / 'HelpCategory.yaml')
    contents = _load_yaml_rows(diff_root / 'HelpContent.yaml')
    category_names = {str(row.get('id') or ''): str(row.get('name') or '') for row in categories}
    entries: list[HelpPageEntry] = []
    for row in contents:
        url = str(row.get('detailUrl') or '').strip()
        if not url:
            continue
        category_id = str(row.get('helpCategoryId') or '')
        entries.append(
            HelpPageEntry(
                help_category_id=category_id,
                help_category_name=category_names.get(category_id, ''),
                item_id=str(row.get('id') or ''),
                name=str(row.get('name') or ''),
                order=int(row.get('order') or 0),
                detail_url=url,
            )
        )
    return entries


def _filter_entries(entries: list[HelpPageEntry], categories: list[str], match: str, limit: int) -> list[HelpPageEntry]:
    category_filter = set(categories)
    match_value = match.casefold().strip()
    filtered: list[HelpPageEntry] = []
    for entry in entries:
        if category_filter and entry.help_category_id not in category_filter:
            continue
        if match_value:
            haystack = '\n'.join(
                (
                    entry.help_category_id,
                    entry.help_category_name,
                    entry.item_id,
                    entry.name,
                    entry.detail_url,
                )
            ).casefold()
            if match_value not in haystack:
                continue
        filtered.append(entry)
        if limit > 0 and len(filtered) >= limit:
            break
    return filtered


def _extract_title_and_text(html: str) -> tuple[str, str]:
    title_match = re.search(r'(?is)<title[^>]*>(.*?)</title>', html)
    title = unescape(title_match.group(1)).strip() if title_match else ''
    main_match = re.search(r'(?is)<main\b[^>]*>(.*?)</main>', html)
    body = main_match.group(1) if main_match else html
    body = re.sub(r'(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>', ' ', body)
    replacements = (
        (r'(?is)<br\s*/?>', '\n'),
        (r'(?is)</p\s*>', '\n\n'),
        (r'(?is)</div\s*>', '\n'),
        (r'(?is)</section\s*>', '\n'),
        (r'(?is)</article\s*>', '\n'),
        (r'(?is)</li\s*>', '\n'),
        (r'(?is)<li[^>]*>', '- '),
        (r'(?is)</h[1-6]\s*>', '\n\n'),
    )
    for pattern, replacement in replacements:
        body = re.sub(pattern, replacement, body)
    body = re.sub(r'(?is)<[^>]+>', ' ', body)
    body = unescape(body)
    body = body.replace('\r\n', '\n').replace('\r', '\n')
    body = re.sub(r'[ \t\f\v]+', ' ', body)
    body = re.sub(r' *\n *', '\n', body)
    body = re.sub(r'\n{3,}', '\n\n', body)
    body = body.strip()
    return title, body


def _safe_name(value: str) -> str:
    return re.sub(r'[^0-9A-Za-z._-]+', '_', value).strip('_') or 'page'


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def _create_ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _is_retryable_fetch_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return False
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, (TimeoutError, ssl.SSLError)):
        return True
    return 'timed out' in str(reason).casefold()


def _fetch_html_bytes_with_curl(url: str, headers: dict[str, str], timeout: float) -> bytes:
    curl_path = shutil.which('curl')
    if not curl_path:
        raise RuntimeError('curl not available')
    command = [
        curl_path,
        '--silent',
        '--show-error',
        '--location',
        '--max-time',
        str(timeout),
    ]
    for key, value in headers.items():
        command.extend(['-H', f'{key}: {value}'])
    command.append(url)
    completed = subprocess.run(command, check=True, capture_output=True)
    return completed.stdout


def _fetch_one(
    entry: HelpPageEntry,
    text_dir: Path,
    timeout: float,
    refresh: bool,
) -> dict[str, Any]:
    file_stub = f'{entry.item_id}_{_safe_name(entry.name)}'
    text_path = text_dir / f'{file_stub}.txt'
    request_headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
    }
    result: dict[str, Any] = {
        'helpCategoryId': entry.help_category_id,
        'helpCategoryName': entry.help_category_name,
        'id': entry.item_id,
        'name': entry.name,
        'order': entry.order,
        'detailUrl': entry.detail_url,
        'textPath': str(text_path),
        'fetchedAt': datetime.now(timezone.utc).isoformat(),
    }
    if text_path.exists() and not refresh:
        text = text_path.read_text(encoding='utf-8', errors='replace')
        result.update(
            {
                'status': 'cached',
                'textLength': len(text),
                'contentSha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            }
        )
        return result
    request = urllib.request.Request(entry.detail_url, headers=request_headers)
    context = _create_ssl_context()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                html_bytes = response.read()
                charset = response.headers.get_content_charset() or 'utf-8'
                html = html_bytes.decode(charset, errors='replace')
                title, text = _extract_title_and_text(html)
                _write_text(text_path, text + '\n')
                result.update(
                    {
                        'status': 'ok',
                        'httpStatus': getattr(response, 'status', None),
                        'title': title or None,
                        'textLength': len(text),
                        'contentSha256': hashlib.sha256(html_bytes).hexdigest(),
                    }
                )
                return result
        except urllib.error.HTTPError as exc:
            result.update({'status': 'error', 'httpStatus': exc.code, 'error': str(exc)})
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if _is_retryable_fetch_error(exc):
                try:
                    html_bytes = _fetch_html_bytes_with_curl(entry.detail_url, request_headers, timeout)
                    html = html_bytes.decode('utf-8', errors='replace')
                    title, text = _extract_title_and_text(html)
                    _write_text(text_path, text + '\n')
                    result.update(
                        {
                            'status': 'ok',
                            'httpStatus': 200,
                            'title': title or None,
                            'textLength': len(text),
                            'contentSha256': hashlib.sha256(html_bytes).hexdigest(),
                        }
                    )
                    return result
                except Exception:
                    pass
            if attempt < 2 and _is_retryable_fetch_error(exc):
                time.sleep(0.5 * (attempt + 1))
                continue
            result.update({'status': 'error', 'error': str(exc)})
            return result
    if last_error is not None:
        result.update({'status': 'error', 'error': str(last_error)})
    return result


def _write_category_markdowns(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    category_dir = output_dir / 'categories'
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        grouped[str(row.get('helpCategoryId') or '')].append(row)
    for category_id, rows in grouped.items():
        category_name = rows[0].get('helpCategoryName') or category_id
        lines = [f'# {category_name} ({category_id})', '']
        for row in sorted(rows, key=lambda item: (item.get('order') or 0, item.get('id') or '')):
            lines.append(f"## {row.get('name') or row.get('id')}")
            lines.append('')
            lines.append(f"- id: `{row.get('id')}`")
            lines.append(f"- url: `{row.get('detailUrl')}`")
            lines.append(f"- status: `{row.get('status')}`")
            if row.get('title'):
                lines.append(f"- title: {row['title']}")
            text_path = row.get('textPath')
            if text_path and Path(text_path).exists():
                text = Path(text_path).read_text(encoding='utf-8')
                lines.append('')
                lines.append('```text')
                lines.append(text.rstrip())
                lines.append('```')
            lines.append('')
        _write_text(category_dir / f'{_safe_name(category_id)}.md', '\n'.join(lines).rstrip() + '\n')


def _write_summary_markdown(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    lines = ['# Help Content Fetch Summary', '']
    lines.append(f'- generated_at: `{datetime.now(timezone.utc).isoformat()}`')
    lines.append(f'- total_pages: {len(manifest)}')
    lines.append('')
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        grouped[str(row.get('helpCategoryId') or '')].append(row)
    for category_id, rows in sorted(grouped.items()):
        category_name = rows[0].get('helpCategoryName') or category_id
        lines.append(f'## {category_name} ({category_id})')
        for row in sorted(rows, key=lambda item: (item.get('order') or 0, item.get('id') or '')):
            lines.append(
                f"- `{row.get('id')}` {row.get('name')} [{row.get('status')}] "
                f"text={row.get('textLength', 0)}"
            )
        lines.append('')
    _write_text(output_dir / 'README.md', '\n'.join(lines).rstrip() + '\n')


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    text_dir = output_dir / 'text'
    entries = _load_entries(root)
    selected = _filter_entries(entries, categories=args.category, match=args.match, limit=args.limit)
    manifest: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_fetch_one, entry, text_dir, args.timeout, args.refresh): entry
            for entry in selected
        }
        for future in as_completed(futures):
            manifest.append(future.result())
    manifest.sort(key=lambda item: (item.get('helpCategoryId') or '', item.get('order') or 0, item.get('id') or ''))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'help_content_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_category_markdowns(output_dir, manifest)
    _write_summary_markdown(output_dir, manifest)
    ok_count = sum(1 for item in manifest if item.get('status') in {'ok', 'cached'})
    error_count = sum(1 for item in manifest if item.get('status') == 'error')
    print(f'wrote manifest: {manifest_path}')
    print(f'pages selected: {len(selected)}')
    print(f'ok_or_cached: {ok_count}')
    print(f'errors: {error_count}')
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
