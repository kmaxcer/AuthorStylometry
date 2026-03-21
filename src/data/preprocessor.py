import re
import os
from pathlib import Path


def load_data(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            data = file.read()
        return data
    except FileNotFoundError:
        return FileNotFoundError
    except Exception:
        return Exception


def save_data(text: str, original_file_path: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(original_file_path)

    output_path = os.path.join(output_dir, file_name)

    try:
        with open(output_path, 'w', encoding="utf-8") as file:
            file.write(text)
        return output_path
    except Exception:
        return Exception


def show_data(text: str, max_length: int = 500):
    if not text:
        return

    preview = text[:max_length] + "..." if len(text) > max_length else text
    print(preview)


def remove_stuttering(text: str) -> str:
    patterns = [
        (r'(\w)-(\1)-(\1)(\w+)', r'\1\4'),

        (r'(\w)-(\1)(\w+)', r'\1\3'),

        (r'(\w)(?:-\1){2,}', r'\1'),

        (r'\b(\w)-(\1)([а-яё]{2,})', r'\1\3'),

        (r'(\w)-(\1)-(\1)([а-яё]+)', r'\1\4'),
    ]

    for i, (pattern, replacement) in enumerate(patterns, 1):
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

    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

    return text


def remove_structural_elements(text: str) -> str:

    lines = text.split('\n')
    cleaned_lines = []

    structure_patterns = [
        r'^\*?\s*ЧАСТЬ\s+(?:ПЕРВАЯ|ВТОРАЯ|ТРЕТЬЯ|ЧЕТВЕРТАЯ|ПЯТАЯ|\d+|[IVXLCDM]+)\s*\*?$',
        r'^\*?\s*ГЛАВА\s+(?:ПЕРВАЯ|ВТОРАЯ|ТРЕТЬЯ|ЧЕТВЕРТАЯ|ПЯТАЯ|\d+|[IVXLCDM]+)\s*\*?$',
        r'^\*?\s*РАЗДЕЛ\s+(?:ПЕРВЫЙ|ВТОРОЙ|ТРЕТИЙ|\d+|[IVXLCDM]+)\s*\*?$',
        r'^\*?\s*ЭПИЛОГ\s*\*?$',
        r'^\*?\s*ЗАКЛЮЧЕНИЕ\s*\*?$',
        r'^\*?\s*ПРОЛОГ\s*\*?$',
        r'^\*?\s*ВСТУПЛЕНИЕ\s*\*?$',
        r'^\*?\s*ПОСЛЕСЛОВИЕ\s*\*?$',
        r'^[IVXLCDM]+\.\s*$',  # I. II. III.
        r'^\d+\.\s*$',  # 1. 2. 3.
        r'^\s*\d+\s*$',  # Номера страниц
        r'^[-—–=_*]{3,}$',  # Декоративные разделители
    ]

    for line in lines:
        line_stripped = line.strip()
        is_structural = False

        for pattern in structure_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                is_structural = True
                break

        if not is_structural:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def normalize_punctuation(text: str) -> str:

    if '—' in text or '–' in text:
        text = text.replace('—', '-').replace('–', '-')

    if '«' in text or '»' in text or '"' in text or '"' in text:
        text = text.replace('«', '"').replace('»', '"')
        text = text.replace('"', '"').replace('"', '"')

    if '…' in text:
        text = text.replace('…', '...')

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
    text = text.strip()

    return text


def process_file(input_file: str, output_dir: str, show_preview: bool = True):
    text = load_data(input_file)
    if not text:
        return False

    if show_preview:
        show_data(text)

    cleaned_text = clean_text(text)

    if show_preview:
        show_data(cleaned_text)

    saved_path = save_data(cleaned_text, input_file, output_dir)

    return bool(saved_path)


def process_multiple_files(input_files: list, output_dir: str):
    for i, file_path in enumerate(input_files, 1):
        process_file(file_path, output_dir, show_preview=False)


def main():
    base_raw = r'C:\Users\admin\PycharmProjects\AuthorStylometry\data\raw'
    base_processed = r'C:\Users\admin\PycharmProjects\AuthorStylometry\data\processed'

    corpora = ['corpus_19_1', 'corpus_19_2']

    for corpus in corpora:
        raw_dir = os.path.join(base_raw, corpus)
        processed_dir = os.path.join(base_processed, corpus)

        os.makedirs(processed_dir, exist_ok=True)

        txt_files = list(Path(raw_dir).glob('*.txt'))


        for file_path in txt_files:

            text = load_data(str(file_path))
            if not text:
                continue

            cleaned_text = clean_text(text)

            save_data(cleaned_text, str(file_path), processed_dir)


if __name__ == "__main__":
    main()