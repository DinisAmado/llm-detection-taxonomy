# Consults 1 results from the database and prints some stats and samples.

python -c "
import sqlite3, json
conn = sqlite3.connect('results/sna.db')
for table in ['results', 'entities', 'relations']:
    count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    print(f'{table}: {count} rows')
sample = conn.execute('SELECT * FROM results LIMIT 2').fetchall()
print(json.dumps(sample, indent=2, default=str))
conn.close()
"

#Consults all results, entities, relations, their taxonomy categories and language 

"
python -c "
import sqlite3
conn = sqlite3.connect('results/sna.db')
print('--- Contagem por tabela ---')
for table in ['results', 'entities', 'relations']:
    count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    print(f'{table}: {count} rows')

print()
print('--- Distribuição por taxonomy_category ---')
for row in conn.execute('SELECT taxonomy_category, COUNT(*) as n FROM results GROUP BY taxonomy_category ORDER BY n DESC'):
    print(f'  {row[0]}: {row[1]}')

print()
print('--- Distribuição por lang ---')
for row in conn.execute('SELECT lang, COUNT(*) as n FROM results GROUP BY lang'):
    print(f'  {row[0]}: {row[1]}')
conn.close()
"