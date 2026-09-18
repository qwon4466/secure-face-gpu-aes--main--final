import os
from pathlib import Path
import sys
import uuid
import pytest
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['TMPDIR']=str(ROOT/'tests/runtime/tmp')
(ROOT/'tests/runtime/tmp').mkdir(parents=True,exist_ok=True)
@pytest.fixture
def work():
    p=ROOT/'tests/runtime/tests'/uuid.uuid4().hex
    p.mkdir(parents=True,mode=0o700)
    return p
