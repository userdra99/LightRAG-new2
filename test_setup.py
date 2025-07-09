import traceback
import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path('.').absolute()))
from lightrag_processor import setup_lightrag

async def test_setup():
    try:
        print(" Testing LightRAG setup...)
