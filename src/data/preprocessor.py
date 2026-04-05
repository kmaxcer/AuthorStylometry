import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        logger.error("Файл не найден: %s", file_path)
        return ""
    try:
        return path.read_text(encoding='utf-8')
    except PermissionError:
        logger.error("Нет доступа к файлу: %s", file_path)
        return ""
    except Exception as e:
        logger.error("Ошибка чтения %s: %s", file_path, e)
        return ""


def save_data(text: str, output_path: str) -> bool:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding='utf-8')
        return True
    except PermissionError:
        logger.error("Нет доступа для записи: %s", output_path)
        return False
    except Exception as e:
        logger.error("Ошибка записи %s: %s", output_path, e)
        return False


def remove_stuttering(text: str) -> str:
    patterns = [
        (r'(\w)-(\1)-(\1)(\w+)', r'\1\4'),
        (r'(\w)-(\1)(\w+)', r'\1\3'),
        (r'(\w)(?:-\1){2,}', r'\1'),
        (r'\b(\w)-(\1)([а-яё]{2,})', r'\1\3'),
        (r'(\w)-(\1)-(\1)([а-яё]+)', r'\1\4'),
    ]
    for pattern, replacement in patterns:
        text, _ = re.subn(pattern, replacement, text)
    return text


def remove_particle_s(text: str) -> str:
    patterns = [
        (r'(\b[а-яА-ЯёЁ]+)-с\b', r'\1'),
        (r'(\b[а-яА-ЯёЁ]+)-с([,.!?;:])\s', r'\1\2 '),
        (r'(\b[а-яА-ЯёЁ]+)-с([.!?]|$)', r'\1\2'),
    ]
    for pattern, replacement in patterns:
        text, _ = re.subn(pattern, replacement, text)
    return text


def remove_ocr_artifacts(text: str) -> str:
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    return text


def remove_structural_elements(text: str) -> str:
    patterns = [
        r'^\*?\s*ЧАСТЬ\s+(?:ПЕРВАЯ|ВТОРАЯ|ТРЕТЬЯ|ЧЕТВЕРТАЯ|ПЯТАЯ|\d+|[IVXLCDM]+)\s*\*?$',
        r'^\*?\s*ГЛАВА\s+(?:ПЕРВАЯ|ВТОРАЯ|ТРЕТЬЯ|ЧЕТВЕРТАЯ|ПЯТАЯ|\d+|[IVXLCDM]+)\s*\*?$',
        r'^\*?\s*РАЗДЕЛ\s+(?:ПЕРВЫЙ|ВТОРОЙ|ТРЕТИЙ|\d+|[IVXLCDM]+)\s*\*?$',
        r'^\*?\s*(?:ЭПИЛОГ|ЗАКЛЮЧЕНИЕ|ПРОЛОГ|ВСТУПЛЕНИЕ|ПОСЛЕСЛОВИЕ)\s*\*?$',
        r'^[IVXLCDM]+\.\s*$',
        r'^\d+\.\s*$',
        r'^\s*\d+\s*$',
        r'^[-—–=_*]{3,}$',
    ]
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not any(p.match(stripped) for p in compiled):
            cleaned.append(line)
    return '\n'.join(cleaned)


def normalize_punctuation(text: str) -> str:
    replacements = {
        '\u2014': '-',
        '\u2013': '-',
        '\u00ab': '"',
        '\u00bb': '"',
        '\u201c': '"',
        '\u201d': '"',
        '\u2026': '...',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = remove_ocr_artifacts(text)
    text = remove_structural_elements(text)
    text = remove_stuttering(text)
    text = remove_particle_s(text)
    text = normalize_punctuation(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?;:…-])', r'\1', text)
    return text.strip()


def process_corpus(raw_dir: str, output_dir: str):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_path.glob('*.txt'))
    if not txt_files:
        logger.warning("Нет .txt файлов в %s", raw_dir)
        return

    ok = 0
    for fp in txt_files:
        text = load_data(str(fp))
        if not text:
            continue
        cleaned = clean_text(text)
        if cleaned and save_data(cleaned, str(out_path / fp.name)):
            ok += 1

    logger.info("%s: обработано %d / %d файлов", raw_dir, ok, len(txt_files))


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'

    for corpus in ['corpus_19_1', 'corpus_19_2']:
        corpus_raw = raw_dir / corpus
        if not corpus_raw.exists():
            logger.warning("Директория не найдена: %s", corpus_raw)
            continue
        process_corpus(str(corpus_raw), str(processed_dir / corpus))


if __name__ == "__main__":
    main()
