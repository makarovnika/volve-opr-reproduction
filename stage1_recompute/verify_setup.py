"""Verify the copied Stage-2 pipeline reproduces the data card (counts + hash)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_volve import build_dataset
from src.config import EXPECTED

d = build_dataset()
c = d["counts"]
print("file_hash:", d["file_hash"])
allok = True
for k, v in EXPECTED.items():
    got = c.get(k, "?")
    ok = (got == v)
    allok &= ok
    print(f"  {k:30s} exp={v:<6} got={got:<6} {'OK' if ok else 'MISMATCH'}")
print("ALL COUNTS MATCH:", allok)
