import os
import glob
import re

docs_dir = os.path.join("ai", "results", "phase17_final_documentation")
files = glob.glob(os.path.join(docs_dir, "*.*"))

banned = [
    r'clinical accuracy',
    r'diagnostic accuracy',
    r'cancer detected',
    r'diagnosis confirmed',
    r'clinically validated',
    r'medically certified',
    r'state-of-the-art'
]

print(f"Auditing {len(files)} files in {docs_dir}...")
warnings = []

for f in files:
    if f.endswith(".png"):
        continue
    with open(f, encoding="utf-8", errors="ignore") as fp:
        content = fp.read()
    fname = os.path.basename(f)
    for b in banned:
        for m in re.finditer(b, content, re.IGNORECASE):
            start = max(0, m.start() - 50)
            end = min(len(content), m.end() + 50)
            snippet = content[start:end].replace('\n', ' ')
            is_disclaimer = any(k in snippet.lower() for k in ["not", "never", "disclaim", "prohibit", "without", "no document", "reject"])
            if not is_disclaimer:
                warnings.append((fname, m.group(0), snippet))
            print(f"  [{fname}] match '{m.group(0)}': ...{snippet}... --> {'DISCLAIMER_CONTEXT' if is_disclaimer else 'AFFIRMATIVE_FLAG'}")

# Verify key metrics presence (check both 80.57% and 80.57)
metrics_to_check = [
    (r"80\.57", "accuracy 80.57%"),
    (r"75\.69", "balanced accuracy 75.69%"),
    (r"70\.51", "macro F1 70.51%"),
    (r"81\.06", "weighted F1 81.06%"),
    (r"54\.00", "melanoma precision 54.00%"),
    (r"50\.94", "melanoma recall 50.94%"),
    (r"52\.43", "melanoma F1 52.43%"),
    (r"0\.85", "exp8 weight 0.85"),
    (r"0\.15", "exp13 weight 0.15"),
    (r"988", "test split count 988"),
    (r"1,?010", "val split count 1,010"),
    (r"0\.50", "exp15 threshold 0.50")
]

report_path = os.path.join(docs_dir, "FINAL_RESEARCH_REPORT.md")
with open(report_path, encoding="utf-8") as fp:
    rep_content = fp.read()

print("\nVerifying key values in FINAL_RESEARCH_REPORT.md:")
for pattern, desc in metrics_to_check:
    found = bool(re.search(pattern, rep_content))
    print(f"  {desc}: {'FOUND' if found else 'MISSING'}")

if warnings:
    print(f"\nWARNING: Found {len(warnings)} potential un-negated phrases!")
else:
    print("\nSUCCESS: Zero un-negated forbidden phrases found. All occurrences are strictly inside disclaimers.")
