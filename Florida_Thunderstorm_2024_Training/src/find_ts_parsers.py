import os, glob

files = set()
for ext in ["*.py"]:
    for f in glob.glob(f"**/{ext}", recursive=True):
        files.add(f)

print(f"Total python files scanned: {len(files)}")
matches = []
for f in sorted(files):
    try:
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            content = fp.read()
            found = []
            if "parse_metar" in content:
                found.append("parse_metar")
            if "has_thunderstorm_ts" in content:
                found.append("has_thunderstorm_ts")
            if 'token in body.split()' in content:
                found.append("token in body.split()")
            if found:
                matches.append((f, found))
    except Exception as e:
        pass

for f, found in matches:
    print(f"{f}: {', '.join(found)}")
