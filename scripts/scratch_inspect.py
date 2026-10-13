import sys
sys.path.insert(0, '.')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
raw = pd.read_csv('data/raw/phenotype.hpoa', sep='\t', comment='#', dtype=str).fillna('')

# What SMA records got merged?
sma_mask = raw['disease_name'].str.lower().str.contains('spinal muscular atrophy', regex=False, na=False)
sma_raw = raw[sma_mask]
ids = sma_raw[['database_id', 'disease_name']].drop_duplicates()
print("=== SMA entries in raw HPOA ===")
for _, r in ids.iterrows():
    cnt = len(sma_raw[sma_raw['database_id'] == r['database_id']])
    print(f"  {r['database_id']:20s} n={cnt:3d}  {r['disease_name'][:70]}")

# Check OMIM:615290 and OMIM:618291 names
print()
print("=== OMIM:615290 ===")
sub = raw[raw['database_id'] == 'OMIM:615290']
print(f"  name: {sub['disease_name'].iloc[0] if not sub.empty else 'N/A'}")
print(f"  n={len(sub)}")
print()
print("=== OMIM:618291 ===")
sub = raw[raw['database_id'] == 'OMIM:618291']
print(f"  name: {sub['disease_name'].iloc[0] if not sub.empty else 'N/A'}")
print(f"  n={len(sub)}")

# Also check Pompe merge - which ids were merged?
pompe_mask = raw['disease_name'].str.lower().str.contains('pompe|glycogen storage disease ii', regex=True, na=False)
pompe_raw = raw[pompe_mask]
ids = pompe_raw[['database_id', 'disease_name']].drop_duplicates()
print()
print("=== Pompe entries in raw HPOA ===")
for _, r in ids.iterrows():
    cnt = len(pompe_raw[pompe_raw['database_id'] == r['database_id']])
    print(f"  {r['database_id']:20s} n={cnt:3d}  {r['disease_name'][:60]}")
