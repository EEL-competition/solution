import os
import path

from datetime import datetime

PROJECT_ROOT = (
    path.Path(os.path.dirname(__file__)).joinpath("..").joinpath("..").abspath()
)
PHRASEOLOGY_ROOT = PROJECT_ROOT.joinpath("phraseology")

DATA_ROOT = PHRASEOLOGY_ROOT.joinpath("cropped_data")
PREPROCESSED_ROOT = PHRASEOLOGY_ROOT.joinpath("preprocessed_data")
LOGS_ROOT = PHRASEOLOGY_ROOT.joinpath("logs")

UTCNOW = datetime.utcnow().strftime("%y%m%d.%H%M%S")
