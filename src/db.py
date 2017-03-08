import psycopg2
from psycopg2 import extras
import numpy as np
import pandas as pd
from collections import Counter


def cursor_connect(cursor_factory=None):
    """
    Connects to the DB and returns the connection and cursor, ready to use.

    :param cursor_factory: psycopg2.extras
    :return: tuple of (psycopg2 connection, psycopg2 cursor)
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

    :param query: string containing SQL SELECT command
    :param curs_dict: dict cursor factory (output is dict)
    :return: list of rows/records (SQL output)
    """
    if curs_dict == True:
        conn, cur = cursor_connect(psycopg2.extras.DictCursor)
    else:
        conn, cur = cursor_connect()
    cur.execute(query)
    rows = cur.fetchall()
    return rows


def feat_icureadm():
    """
    Function extracts the counts of ICU readmissions for each patient as features.

    :return: pandas DataFrame
    """
    # query
    q_icupat = """SELECT * FROM
        (SELECT subject_id, COUNT(icustay_id) AS n_icustays
        FROM icustays
        GROUP BY subject_id) AS sub_q
    WHERE n_icustays > 1;"""

    # Query output
    icupat = exec_query(q_icupat, False)
    df_icupat = pd.DataFrame(icupat, columns=['subjectid', 'n_icustays'])

    n_readm = pd.Series(df_icupat.n_icustays - 1, name='n_readm')
    df_icu = pd.concat([df_icupat.subjectid, n_readm], axis=1)
    return df_icu


def feat_trav():
    """
    Function extracts the count of traversals for all wards and only ICU wards for each patient's hospital admission as
    features.

    :return: tuple of (merged pandas DataFrame, main pandas DataFrame)
    """
    left_df = feat_icureadm()

    q_mult = """SELECT subject_id, hadm_id, icustay_id, eventtype,
    prev_careunit, curr_careunit, prev_wardid, curr_wardid, intime, outtime, los
    FROM transfers;"""
    mult_trav = exec_query(q_mult, False)
    mult_col = ['subjectid', 'hadmid', 'icustayid', 'eventtype', 'prev_cu', 'curr_cu',
                'prev_wardid', 'curr_wardid', 'intime', 'outtime', 'los']
    df_mult = pd.DataFrame(mult_trav, columns=mult_col)
    df_mult.replace(to_replace='', value=np.nan, inplace=True, regex=True)

    # filter for ICU patients with readmissions
    filter_preadm = list(left_df.subjectid)
    df_mult_readm = df_mult[df_mult.subjectid.isin(filter_preadm)] # main DF

    # filter for exclusion of neonate patients
    df_mult_readm = df_mult_readm[df_mult_readm['prev_cu'] != 'NWARD']
    df_mult_readm = df_mult_readm[df_mult_readm['prev_cu'] != 'NICU']
    df_mult_readm = df_mult_readm[df_mult_readm['curr_cu'] != 'NWARD']
    df_mult_readm = df_mult_readm[df_mult_readm['curr_cu'] != 'NICU']

    # extract feature: n_trav
    df_mult_readm_grp = df_mult_readm.groupby(['subjectid', 'hadmid']).size()
    df_mult_readm_grp = df_mult_readm_grp.to_frame(name='n_trav').reset_index()

    # join DF on subjectid to add n_icustays col
    df_icu1 = pd.merge(df_mult_readm_grp, left_df, on='subjectid',
                       how='left')

    # extract feature: n_icutrav
    df_mult_readm_icu = df_mult_readm[df_mult_readm.icustayid.notnull() == True]
    df_mult_readm_hadm = df_mult_readm_icu.groupby(['subjectid', 'hadmid']).size().to_frame('n_icutrav').reset_index()

    # join DF  on subjectid to add n_readm col
    df_icu2 = pd.merge(df_icu1, df_mult_readm_hadm.loc[:, ['hadmid', 'n_icutrav']],
                       on='hadmid', how='inner')
    return (df_icu2, df_mult_readm_icu)


def feat_icustay():
    """
    Function extracts the count of unique ICU stays for each patient's hospital admission as features.
    :return: pandas DataFrame
    """
    left_df = feat_trav()[0]

    # query
    q_icustay = """SELECT subject_id, hadm_id, COUNT(DISTINCT icustay_id)
    FROM transfers
    GROUP BY subject_id, hadm_id;
    """

    # Query output
    icustay = exec_query(q_icustay, False)
    df_icustay = pd.DataFrame(icustay, columns=['subjectid', 'hadmid',
                                                'n_icustays'])

    # join DF  on subjectid to add n_readm col
    df_icu3 = pd.merge(left_df, df_icustay.loc[:, ['hadmid', 'n_icustays']],
                       on='hadmid', how='inner')
    return df_icu3


def feat_transpairs():
    """

    :return:
    """
    left_df = feat_icustay()
    main_df = feat_trav()[1]

    # Identify transfer pairs
    df_trav_copy = main_df.copy() # copy
    df_trav_copy.prev_cu.fillna('nonicu', inplace=True)
    df_trav_copy.curr_cu.fillna('nonicu', inplace=True)
    df_trav_copy['trans'] = df_trav_copy.prev_cu + '-' + df_trav_copy.curr_cu # transfer pairs

    df_toppairs = df_trav_copy.trans.value_counts(ascending=False).to_frame() # count of pairs
    df_top = df_toppairs.transpose().iloc[:, 0:11]  # transpose to columns

    # Pair counter
    sid = list(df_trav_copy.subjectid.value_counts().index) # unique subject_id

    main_d = dict()
    for subj in sid:
        pair_d = dict(Counter(df_trav_copy[df_trav_copy.subjectid==subj].trans))
        pair_d['subjectid'] = subj # add subjectid key
        main_d[subj] = pair_d

    df_pairct = pd.DataFrame.from_dict(main_d, orient='index')

    # drop non-top trans pair cols
    pairs_drop = list(df_toppairs.iloc[10:].index)
    df_pairct.drop(pairs_drop, axis=1, inplace=True)

    df_icu4 = pd.merge(left_df, df_pairct, on='subjectid', how='left')
    return df_icu4


def feat_iculos():
    """
    Function extracts the average length of stay for each patient's ICU stay as a feature, extracting the information
    using the main DataFrame (pre-grouped).

    :return: merged pandas DataFrame
    """
    left_df = feat_transpairs()
    main_df = feat_trav()[1]

    avgiculos = main_df.groupby(['subjectid', 'hadmid'])['los'].mean()
    df_avgiculos = avgiculos.to_frame(name='avg_iculos').reset_index()

    # Merge
    df_icu_final = pd.merge(left_df, df_avgiculos.loc[:, ['hadmid', 'avg_iculos']], on='hadmid', how='left')
    # df_icu5.groupby(['subjectid'])['avg_iculos'].mean() # overall LOS
    return df_icu_final


if __name__ == "__main__":
    pass
