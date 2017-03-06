import psycopg2
from psycopg2 import extras


def cursor_connect(cursor_factory=None):
    """
    Connects to the DB and returns the connection and cursor, ready to use.
    Parameters
    ----------
    cursor_factory : psycopg2.extras
    Returns
    -------
    (psycopg2.extensions.connection, psycopg2.extensions.cursor)
        A tuple of (psycopg2 connection, psycopg2 cursor).
    """
    #DB connection
    conn = psycopg2.connect(dbname="mimic", user="mimic", host="localhost", port="2345",
                            password="oNuemmLeix9Yex7W")
    if not cursor_factory:
        cur = conn.cursor()
    else:
        cur = conn.cursor(cursor_factory=cursor_factory)
    return conn, cur


def exec_query(query, curs_dict=True):
    """
    Execute query and returns the SQL output.

    Parameters
    ----------
    query: string containing SQL SELECT command
    curs_dict: dict cursor factory (output is dict)

    Returns
    -------
    rows: list of rows/records (SQL output)
    """
    if curs_dict == True:
        conn, cur = cursor_connect(psycopg2.extras.DictCursor)
    else:
        conn, cur = cursor_connect()
    cur.execute(query)
    rows = cur.fetchall()
    return rows


if __name__ == "__main__":
    # query = """SELECT subject_id, hadm_id FROM admissions LIMIT 10;"""
    # # conn, cur = cursor_connect()
    # conn, cur = cursor_connect(psycopg2.extras.DictCursor)  # access retrieved records as Python dict using keys
    # cur.execute(query)
    # rows = cur.fetchall()
    # # print rows[0]['subject_id']
    # for row in rows:
    #     print row