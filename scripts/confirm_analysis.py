import sys, csv, json, os
sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIRM 1: unit of scoring and item count for end-to-end accuracy ---
heldout_claims = r'c:\Users\Gowthami\TruthSync\dataset\layer5_heldout\heldout_claims.csv'
heldout_resp   = r'c:\Users\Gowthami\TruthSync\dataset\layer5_heldout\heldout_responses.csv'

print("=== CONFIRM 1: End-to-end scoring unit and item count ===")
if os.path.exists(heldout_claims):
    with open(heldout_claims, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f"  heldout_claims.csv rows: {len(rows)}")
    labels = set(r.get('label','') for r in rows)
    print(f"  Labels: {labels}")
    rids = set(r.get('response_id','') for r in rows)
    print(f"  Unique response_ids in claims: {len(rids)}")
else:
    print("  heldout_claims.csv NOT FOUND")

if os.path.exists(heldout_resp):
    with open(heldout_resp, encoding='utf-8') as f:
        rows_r = list(csv.DictReader(f))
    print(f"  heldout_responses.csv rows: {len(rows_r)}")
    verdicts = set(r.get('response_verdict','') for r in rows_r)
    print(f"  Unique verdicts: {verdicts}")
else:
    print("  heldout_responses.csv NOT FOUND")

# --- CONFIRM 2: training stage and data for template comparison ---
print()
print("=== CONFIRM 2: Template comparison training stage ===")
train_csv = r'c:\Users\Gowthami\TruthSync\dataset\layer1_nli\train.csv'
val_csv   = r'c:\Users\Gowthami\TruthSync\dataset\layer1_nli\val.csv'
test_csv  = r'c:\Users\Gowthami\TruthSync\dataset\layer1_nli\test.csv'
for path in [train_csv, val_csv, test_csv]:
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            rows2 = list(csv.DictReader(f))
        print(f"  {os.path.basename(path)}: {len(rows2)} rows")
    else:
        print(f"  {os.path.basename(path)}: NOT FOUND")

# Also check models directory for training logs
models_dir = r'c:\Users\Gowthami\TruthSync\models'
print(f"  models/ contents: {os.listdir(models_dir) if os.path.exists(models_dir) else 'NOT FOUND'}")

# --- CONFIRM 3: Retrieval measured before or after reranking ---
print()
print("=== CONFIRM 3: Retrieval metrics measured before or after reranking ===")
retrieval_test = r'c:\Users\Gowthami\TruthSync\dataset\layer2_retrieval\retrieval_test.csv'
if os.path.exists(retrieval_test):
    with open(retrieval_test, encoding='utf-8') as f:
        rows3 = list(csv.DictReader(f))
    print(f"  retrieval_test.csv rows: {len(rows3)}")
    cols = rows3[0].keys() if rows3 else []
    print(f"  Columns: {list(cols)}")
    # Count unique claims
    cids = set(r.get('claim_id','') for r in rows3)
    print(f"  Unique claim_ids: {len(cids)}")
    # Check for rerank_score column
    if 'rerank_score' in cols:
        print("  Has rerank_score column -> metrics measured AFTER reranking")
    else:
        print("  No rerank_score column -> metrics measured BEFORE/AT retrieval stage")
else:
    print("  retrieval_test.csv NOT FOUND")

# Check src/retrieval for how retrieval evaluation works
eval_scripts = []
for root, dirs, files in os.walk(r'c:\Users\Gowthami\TruthSync\src'):
    for f in files:
        if 'eval' in f.lower() or 'retriev' in f.lower():
            eval_scripts.append(os.path.join(root, f))
print(f"  Retrieval-related src files: {eval_scripts}")

for root, dirs, files in os.walk(r'c:\Users\Gowthami\TruthSync\scripts'):
    for f in files:
        if 'eval' in f.lower() or 'retriev' in f.lower():
            eval_scripts.append(os.path.join(root, f))
print(f"  Retrieval-related script files: {eval_scripts}")

# --- CONFIRM 4: Which evaluation set produced the 52-error log ---
print()
print("=== CONFIRM 4: Evaluation set from which the 52-error log was drawn ===")
error_log_paths = []
for root, dirs, files in os.walk(r'c:\Users\Gowthami\TruthSync'):
    for f in files:
        if 'error' in f.lower() or 'log' in f.lower() or 'fail' in f.lower():
            error_log_paths.append(os.path.join(root, f))
print(f"  Error log files found: {error_log_paths[:15]}")

# check logs directory
logs_dir = r'c:\Users\Gowthami\TruthSync\logs'
if os.path.exists(logs_dir):
    print(f"  logs/ contents: {os.listdir(logs_dir)}")
else:
    print("  logs/ directory not found")

results_dir = r'c:\Users\Gowthami\TruthSync\results'
if os.path.exists(results_dir):
    print(f"  results/ contents: {os.listdir(results_dir)}")
else:
    print("  results/ directory not found")
