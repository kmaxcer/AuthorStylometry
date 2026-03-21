import json
from pathlib import Path
from typing import List


class WindowGenerator:

    def __init__(self, window_size: int = 1000, step: int = 500, min_window_size: int = 200):
        self.window_size = window_size
        self.step = step
        self.min_window_size = min_window_size
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.processed_dir = self.base_dir / 'data' / 'processed'
        self.windows_dir = self.base_dir / 'data' / 'windows'

    def load_text(self, file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return ""

    def create_windows(self, text: str) -> List[str]:
        windows = []
        text_length = len(text)

        if text_length <= self.window_size:
            if text_length >= self.min_window_size:
                windows.append(text)
            return windows

        start = 0
        while start < text_length - self.min_window_size:
            end = min(start + self.window_size, text_length)
            window = text[start:end]

            if len(window) >= self.min_window_size:
                windows.append(window)

            start += self.step

        return windows

    def save_window(self, window: str, output_path: Path) -> bool:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(window)
            return True
        except:
            return False

    def process_file(self, file_path: Path, corpus_name: str) -> List[dict]:
        file_metadata = []
        text = self.load_text(file_path)

        if not text:
            return file_metadata

        windows = self.create_windows(text)

        if not windows:
            return file_metadata

        corpus_windows_dir = self.windows_dir / corpus_name
        corpus_windows_dir.mkdir(parents=True, exist_ok=True)

        for i, window in enumerate(windows):
            window_name = f"{file_path.stem}_w{i:04d}.txt"
            window_path = corpus_windows_dir / window_name

            if self.save_window(window, window_path):
                file_metadata.append({
                    'file_id': file_path.stem,
                    'corpus': corpus_name,
                    'author': file_path.stem.split('_')[0],
                    'window_id': i,
                    'window_file': window_name,
                    'window_size': len(window)
                })

        return file_metadata

    def process_corpus(self, corpus_name: str) -> List[dict]:
        corpus_path = self.processed_dir / corpus_name

        if not corpus_path.exists():
            return []

        txt_files = list(corpus_path.glob('*.txt'))
        all_metadata = []

        for file_path in txt_files:
            metadata = self.process_file(file_path, corpus_name)
            all_metadata.extend(metadata)

        return all_metadata

    def save_metadata(self, metadata: List[dict], corpus_name: str):
        if not metadata:
            return

        metadata_path = self.windows_dir / corpus_name / 'windows_metadata.json'

        metadata_with_stats = {
            'corpus': corpus_name,
            'window_size': self.window_size,
            'step': self.step,
            'total_windows': len(metadata),
            'windows': metadata
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_with_stats, f, ensure_ascii=False, indent=2)

    def run(self, corpora: List[str] = None):
        if corpora is None:
            corpora = ['corpus_19_1', 'corpus_19_2']

        self.windows_dir.mkdir(parents=True, exist_ok=True)

        for corpus_name in corpora:
            metadata = self.process_corpus(corpus_name)
            if metadata:
                self.save_metadata(metadata, corpus_name)


def main():
    generator = WindowGenerator(window_size=1000, step=500, min_window_size=200)
    generator.run()


if __name__ == "__main__":
    main()