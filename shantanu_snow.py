import snowflake.connector
import time

ctx = snowflake.connector.connect(
    user='SHANTANU',
    password='Sgnp9999@',
    account='mxvkytv-kfb20355',
    warehouse = 'COMPUTE_WH',
    database = 'SHANTANU_MISTRAL',
    schema = 'MISTRAL'
    )
cs = ctx.cursor()

def snowflake_run(query):
    cs.execute(f"{query}")
    rows = cs.fetchall()
    for row in rows:
        time.sleep(10)
        result = row[0]
    return result

def snowflake_run_new(query):
    cs.execute(f"{query}")
    rows = cs.fetchall()
    for row in rows:
        result = row[0]
    return "Done"