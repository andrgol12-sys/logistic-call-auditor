"""Local integration check: real PostgreSQL, no Telegram/STT/LLM calls."""
import os
import subprocess
import sys
from pathlib import Path
import bot
import psycopg2

subprocess.run([sys.executable, 'test_offline.py'], check=True)
assert bot.load_audit_prompt(Path('prompt.md'))
try:
    bot.Settings.from_env()
except ValueError as error:
    assert 'TELEGRAM_BOT_TOKEN' in str(error)
else:
    raise AssertionError('Missing keys must be rejected')
print('Real module import, prompt loading, missing-key validation: OK', flush=True)
url = os.environ['DATABASE_URL']
bot.init_database(url)
bot.save_audit_result(url, 0, None, None, 'offline_fixture', 'offline_fixture',
                      'synthetic.txt', 'SYNTHETIC TRANSCRIPT', 'SYNTHETIC ANALYSIS',
                      'offline_test', None)
with psycopg2.connect(url) as connection:
    with connection.cursor() as cursor:
        cursor.execute('SELECT transcript, analysis, status FROM video_audits WHERE file_id=%s',
                       ('offline_fixture',))
        assert cursor.fetchone() == ('SYNTHETIC TRANSCRIPT', 'SYNTHETIC ANALYSIS', 'offline_test')
print('Real PostgreSQL schema, insert and readback: OK (synthetic fixture)', flush=True)
print('NO TELEGRAM / ASSEMBLYAI / OPENAI CALLS', flush=True)
