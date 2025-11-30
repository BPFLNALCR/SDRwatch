from pathlib import Path
text = Path(''sdrwatch.py'').read_text()

def rb(src, start, end):
    s = src.index(start)
    e = src.index(end, s)
    return src[:s] + src[e:]

text = rb(text, '# ------------------------------\n# Utility\n# ------------------------------\n\n', '@dataclass\nclass Segment')
text = rb(text, '@dataclass\nclass Segment', '@dataclass\nclass ScanProfile')
text = rb(text, 'class Store:', '# ------------------------------\n# SDR sources\n')
text = rb(text, '# ------------------------------\n# CFAR helpers\n', 'def _track_spur_hits')
text = rb(text, '# ------------------------------\n# Output helpers\n', 'class ScanLogger:')
Path(''sdrwatch.py'').write_text(text)
