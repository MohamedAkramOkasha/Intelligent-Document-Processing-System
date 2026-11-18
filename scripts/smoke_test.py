"""Run every sample through the pipeline (no server needed) and print a summary.

    python scripts/smoke_test.py [samples_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.processing.pipeline import DocumentPipeline  # noqa: E402


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    pipeline = DocumentPipeline()
    print(f"Extractor: {pipeline.extractor.name}\n")
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        out = pipeline.process(path.read_bytes(), path.name)
        r = out.result
        print(f"== {path.name}: {r.document_type.value} (cls {r.classification_confidence:.2f}) "
              f"confidence={r.overall_confidence:.2f} review={r.needs_review} [{r.processing_time_ms} ms]")
        for reason in r.review_reasons:
            print(f"   - {reason}")
        if "-v" in sys.argv:
            print(json.dumps(r.data, indent=2, default=str))


if __name__ == "__main__":
    main()
