import json
from pathlib import Path

from datalab_sdk import DatalabClient
from datalab_sdk.models import ProcessingOptions

underlay_path = Path(__file__).parent
path = Path(__file__).parent / "docs" / "image.png"


client = DatalabClient(api_key="G1pw15-dT5jRTmlPZAjUK-ZvbPCmoRQQfbQOLAlft7U")
client.ocr(path, save_output=Path(__file__).parent / "output")
with open(Path(__file__).parent / "ocr_result_pass.json", "w") as f:
    json.dump(
        client.ocr(path, ProcessingOptions(), save_output=Path(__file__).parent),
        f,
        indent=4,
        ensure_ascii=False,
    )
