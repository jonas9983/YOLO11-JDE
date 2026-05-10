import sqlite3
import difflib

conn = sqlite3.connect('npc_database.db')
cursor = conn.cursor()

# Get all unique athlete names
cursor.execute("SELECT DISTINCT athlete_name FROM athletes WHERE athlete_name NOT LIKE '%COMPARISONS%' AND athlete_name NOT LIKE '%AWARDS%'")
names = [row[0] for row in cursor.fetchall() if row[0] and len(row[0]) > 3]

# Simple fuzzy matching to find aliases
print(f"Total unique names to process: {len(names)}")
print("Finding potential aliases (this might take a few seconds)...")

aliases = []
# We'll just check a subset for demonstration to be fast
for i, name1 in enumerate(names[:500]):
    # Find close matches in the rest of the list
    matches = difflib.get_close_matches(name1, names[i+1:], n=3, cutoff=0.9)
    if matches:
        for match in matches:
            aliases.append((name1, match))

print("\n--- Examples of Discovered Aliases ---")
for a, b in aliases[:20]:
    print(f"'{a}'  <--->  '{b}'")
