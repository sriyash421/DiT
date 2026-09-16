"""Net effect of running the chain: how many captions does it fix, and how many does it break?

pass@k can only rise, so it cannot show breakage. This counts, per caption, whether attempt 1 and
attempt W are exact, and reports the two directions separately.
"""
import json

T = json.load(open("/gscratch/socialrl/sriyash/clevr_g6_bon/traces/control_5000/traces.json"))
for W in (2, 4, 8):
    a1 = [t["attempts"][0]["exact"] for t in T]
    aW = [t["attempts"][W - 1]["exact"] for t in T]
    gained = sum(1 for x, y in zip(a1, aW) if not x and y)
    lost = sum(1 for x, y in zip(a1, aW) if x and not y)
    ever = sum(1 for t in T if any(a["exact"] for a in t["attempts"][:W]))
    print(f"window {W}: exact at attempt 1 = {sum(a1):3d}/200   at attempt {W} = {sum(aW):3d}/200")
    print(f"           fixed {gained:3d}   broken {lost:3d}   net {gained - lost:+d}"
          f"   (ever exact within the window: {ever})")
