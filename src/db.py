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


def exec_query(query, curs_dict=False):
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
    Function extracts unique ICU readmission stays as a record and their respective associated patients (subject_id),
    admission time, discharge time, and information related to the previous ICU stay.

    :return: pandas DataFrame
    """
    # Patients and their Number of Days until a Unique ICU Readmission
    q_icustay = """SELECT * FROM
    (SELECT subject_id, icustay_id, min_in, max_out,
    min_in - lag(max_out)
    OVER (PARTITION BY subject_id ORDER BY min_in) AS diff
    FROM
    (SELECT subject_id, icustay_id,
    MIN(intime) as min_in, MAX(outtime) AS max_out
    FROM transfers
    WHERE icustay_id IS NOT NULL
    GROUP BY subject_id, icustay_id) as sub_q
    ORDER BY subject_id) as sub
    WHERE diff is not null;"""

    icustay = exec_query(q_icustay) # query output
    df_icustay_time = pd.DataFrame(icustay, columns=['subjectid', 'icustayid',
                                                     'icu_intime',  # first unique ICU admission time
                                                     'icu_outtime',  # unique ICU discharge time
                                                     'readm_days'])  # number of days since last ICU discharge/transfer
    df_icustay_time[
        'icu_prev_outtime'] = df_icustay_time.icu_intime - df_icustay_time.readm_days  # time of previous ICU discharge/transfer

    df_icustay_time.readm_days = np.round(df_icustay_time.readm_days.astype('int') * (1 / 8.64e13), 3)

    # Previous ICU stay ID
    q_previcu = """SELECT DISTINCT ON (subject_id, icustay_id, outtime) subject_id, icustay_id, outtime
    FROM transfers
    WHERE icustay_id IS NOT NULL;"""

    prev_icustay = exec_query(q_previcu) # query output
    df_previcu = pd.DataFrame(prev_icustay, columns=['subjectid', 'prev_icustayid', 'icu_prev_outtime'])

    df_icustay = pd.merge(df_icustay_time, df_previcu, on=['subjectid', 'icu_prev_outtime'], how='left')
    df_icustay.drop(labels='icu_prev_outtime', axis=1, inplace=True)
    return df_icustay


def exclusion(DataFrame):
    """
    Function filters the DataFrame to exclude Neonatal and minor patients (subject_id with age < 18 years old).

    :param DataFrame: ICU readmission pandas DataFrame
    :return: pandas DataFrame filtered by exclusion criteria
    """
    # Neonates
    q_nicu = """SELECT DISTINCT icustay_id FROM transfers
    WHERE curr_careunit = 'NICU' AND icustay_id IS NOT NULL;
    """
    nicu_stays = exec_query(q_nicu) # query output
    df_nicu_stays = pd.DataFrame(nicu_stays, columns=['icustayid'])

    df_icustay = DataFrame[DataFrame['icustayid'].isin(df_nicu_stays.icustayid) == False]

    # Minors
    # age of patients < 90
    q_age_hadm1 = """SELECT a.subject_id,
        FLOOR((a.admittime::date - p.dob::date)/365.0) AS age
        FROM admissions as a
        INNER JOIN patients as p
        ON a.subject_id = p.subject_id
        WHERE FLOOR((a.admittime::date - p.dob::date)/365.0) < 90;"""

    # adjusted age of patients > 89
    q_age_hadm2 = """SELECT a.subject_id,
        FLOOR((a.admittime::date - p.dob::date)/365.0) -210 AS age
        FROM admissions as a
        INNER JOIN patients as p
        ON a.subject_id = p.subject_id
        WHERE FLOOR((a.admittime::date - p.dob::date)/365.0) > 89;"""

    age_hadm1 = exec_query(q_age_hadm1, False)
    age_hadm2 = exec_query(q_age_hadm2, False)
    df_age_hadm1 = pd.DataFrame(age_hadm1, columns=['subjectid', 'age'])
    df_age_hadm2 = pd.DataFrame(age_hadm2, columns=['subjectid', 'age'])

    df_age_hadm = pd.concat([df_age_hadm1, df_age_hadm2])
    df_adults = df_age_hadm[df_age_hadm.age > 17]
    df_adults_sid = list(df_adults.subjectid.value_counts().index.sort_values())

    df_excluded = df_icustay[df_icustay.subjectid.isin(df_adults_sid)]
    return df_excluded


def feat_prev_iculos(DataFrame):
    """
    Function extracts the total length of stay (LOS) of the previous unique ICU stay, inclusive of the duration of all
    intra-ICU stays.

    :param DataFrame: pandas DataFrame
    :return: pandas DataFrame + previous ICU length of stay feature
    """
    q_prevlos = """SELECT icustay_id, los
    FROM icustays;"""

    prevlos = exec_query(q_prevlos) # query output
    df_prevlos = pd.DataFrame(prevlos, columns=['prev_icustayid', 'prev_iculos'])

    df_previculos = pd.merge(DataFrame, df_prevlos, on='prev_icustayid', how='left')
    return df_previculos


def feat_intra_trans(DataFrame):
    """
    Function extracts the number of non-unique, intra-ICU ward transfers for each patient's unique ICU stay.

    :param DataFrame: pandas DataFrame
    :return: pandas DataFrame + intra_trans feature
    """
    q_multtrav = """SELECT icustay_id, COUNT(*)
    FROM transfers
    WHERE icustay_id IS NOT NULL
    GROUP BY icustay_id"""

    mult_trav = exec_query(q_multtrav) # query output
    df_multtrav = pd.DataFrame(mult_trav, columns=['icustayid', 'n_icutrav'])
    df_intratrans = pd.merge(DataFrame, df_multtrav, on='icustayid', how='left')
    return df_intratrans


def binary_cu(careunit):
    """
    Helper function takes the careunit type and returns a binary value indicating whether the careunit was of an ICU
    type or not:
      1. 0: non-ICU
      2: 1: ICU

    :param careunit: ward type
    :return: binary value indicating ICU
    """
    if careunit > 0 and careunit < 7:
        x = 1
    else:
        x = 0
    return x


def feat_curr_careunit(DataFrame):
    """
    Function extracts three features regarding the type of ICU the patient was admitted or transferred into:
      1. prev_cu: categorical feature indicating the previous care unit
      2. curr_cu: categorial feature indicating the current care unit
      3. prev_ICU: binary feature indicating previous ICU

    :param DataFrame: panda DataFrame
    :return: pandas DataFrame + prev_cu, curr_cu, prev_ICU features
    """
    q_careunit = """SELECT DISTINCT ON (icustay_id, intime) icustay_id, intime, curr_careunit, prev_careunit
    FROM transfers
    WHERE icustay_id IS NOT NULL;"""

    careunit = exec_query(q_careunit) # query output
    df_careunit = pd.DataFrame(careunit, columns=['icustayid', 'icu_intime', 'curr_cu', 'prev_cu'])

    # Replace categorical care unit strings with integers
    df_careunit.prev_cu.replace(to_replace=
                                {'': 0, 'MICU': 1, 'CSRU': 2, 'SICU': 3, 'CCU': 4, 'TSICU': 5, 'NICU': 6, 'NWARD': 7},
                                inplace=True)
    df_careunit.curr_cu.replace(to_replace=
                                {'': 0, 'MICU': 1, 'CSRU': 2, 'SICU': 3, 'CCU': 4, 'TSICU': 5, 'NICU': 6, 'NWARD': 7},
                                inplace=True)

    # Binarize if Care Unit was of an ICU type
    df_careunit['prev_ICU'] = df_careunit.prev_cu.apply(binary_cu)
    # df_careunit['curr_ICU'] = df_careunit.curr_cu.apply(binary_cu)

    df_currcu = pd.merge(DataFrame, df_careunit, on=['icustayid', 'icu_intime'], how='left')
    return df_currcu


def feat_disch_careunit(DataFrame):
    """
    Function extracts two features regarding the care unit/ward that the patient was discharged to from the given ICU
    stay.
      1. disch_cu: categorical feature indicating the discharge unit from ICU
      2. disch_ICU: binary feature indicating whether the discharge unit was of an ICU type

    :param DataFrame: pandas DataFrame
    :return: pandas DataFrame + dishc_cu, disch_ICU features
    """
    q_disch = """SELECT DISTINCT ON (t1.outtime) t1.subject_id, t1.icustay_id, t2.curr_careunit, t1.outtime
    FROM
      (SELECT * FROM transfers WHERE curr_careunit LIKE '%U') as t1
    INNER JOIN
      (SELECT * FROM transfers WHERE prev_careunit != '') as t2
    ON t1.outtime = t2.intime"""

    disch_unit = exec_query(q_disch) # query output
    df_disch = pd.DataFrame(disch_unit,
                            columns=['subjectid', 'icustayid', 'disch_cu', 'icu_outtime'])

    # Replace categorical care unit strings with integers
    df_disch['disch_cu'].replace(to_replace=
                                 {'': 0, 'MICU': 1, 'CSRU': 2, 'SICU': 3, 'CCU': 4,
                                  'TSICU': 5, 'NICU': 6, 'NWARD': 7}, inplace=True)

    # Binarize if Care Unit was of an ICU type
    df_disch['disch_ICU'] = df_disch.disch_cu.apply(binary_cu)


    df_dischcu = pd.merge(DataFrame, df_disch[['icustayid', 'disch_cu', 'icu_outtime', 'disch_ICU']],
                           on=['icustayid', 'icu_outtime'], how='inner')
    return df_dischcu


def day_night(datetime):
    """
    Helper function taking the time and returning a binary value indicating whether the time of event was during the
    day or not (night):
      1. 0: night (6:00 PM - 6:00 AM)
      2. 1: day (6:00 AM - 6:00 PM)

    :param datetime: NumPy datetime64 value
    :return: categorical binary value
    """
    hour = np.timedelta64(np.datetime64(datetime, 'h') - (np.datetime64(datetime, 'D')), 'h')
    if hour.astype(np.int64) >=6 and hour.astype(np.int64) <=18:
        time = 1 # day
    else:
        time = 0 # night
    return time


def feat_stay_time(DataFrame):
    """
    Function extracts two binary features indicating whether the time of ICU stay admission and discharge were during
    the day or not (night).
      1. icu_in_day: time of ICU admission
      2. icu_out_day: time of ICU discharge

    Legend:
      0: night
      1: day

    :param DataFrame: pandas DataFrame
    :return: pandas DataFrame + stay_time features
    """
    DataFrame['icu_in_day'] = DataFrame['icu_intime'].apply(day_night)
    DataFrame['icu_out_day'] = DataFrame['icu_outtime'].apply(day_night)
    return DataFrame


def intra_interval(d_risk, stay):
    """
    Helper function mapping the risk score to its respective icustay_id.

    :param d_risk: dictionary of risk score mapped to icustay_id
    :param stay: ICU stay identification
    :return: dictionary risk score value mapped to the subject_id and icustay_id
    """
    risk_score = str.split(stay, '-')
    sid = int(risk_score[0])
    stayid = int(risk_score[1])
    return d_risk[sid][stayid]


def feat_riskscore_intraint(DataFrame):
    """
    Function compute the risk scores of intra-period ICU readmissions (multiple unique ICU admissions within the
    specified interval).

    Risk score = number of icu readmissions = number of total ICU admissions - 1

    The value of the risk score indicates the count/frequency of readmissions for the given ICU stay.

    :param DataFrame: pandas DataFrame
    :return: pandas DataFrame + intra-interval readmission risk score feature
    """
    d_risk = dict()
    for i, row in DataFrame.iterrows():
        if d_risk.has_key(row.subjectid):
            d_risk[row.subjectid]['count'] += 1
            d_risk[row.subjectid][row.icustayid] = d_risk[row.subjectid]['count']

        else:
            d_icu = {'count': 0}
            d_icu[row.icustayid] = d_icu['count']
            d_risk[row.subjectid] = d_icu

    DataFrame['intra_risk'] = DataFrame.subjectid.astype(str) + '-' + DataFrame.astype(str).icustayid
    DataFrame['intra_risk'] = DataFrame['intra_risk'].apply((lambda x: intra_interval(d_risk=d_risk, stay=x)))
    return DataFrame

def trav_pairs():
    """
    Helper function extracting the top 10 traversal pairs of ICU care units.

    :return: tuple of pandas DataFrame
    """
    # Traversal Pairs
    q_trav = """SELECT subject_id, icustay_id, eventtype,
    prev_careunit, curr_careunit
    FROM transfers
    WHERE icustay_id IS NOT NULL;"""

    mult_trav = exec_query(q_trav, False)  # query output
    mult_col = ['subjectid', 'icustayid', 'eventtype', 'prev_cu', 'curr_cu']
    df_trav = pd.DataFrame(mult_trav, columns=mult_col)
    df_trav.replace(to_replace='', value=np.nan, inplace=True, regex=True)

    # Filter for neonate wards
    df_trav = df_trav[df_trav.prev_cu != 'NICU']
    df_trav = df_trav[df_trav.prev_cu != 'NWARD']
    df_trav = df_trav[df_trav.curr_cu != 'NICU']
    df_trav = df_trav[df_trav.curr_cu != 'NWARD']
    df_trav.prev_cu.fillna('nonicu', inplace=True)
    df_trav.curr_cu.fillna('nonicu', inplace=True)

    df_trav['trans'] = df_trav.prev_cu + '-' + df_trav.curr_cu

    # Filter for Patients with ICU readmission
    q_icupat = """SELECT * FROM
        (SELECT subject_id, COUNT(icustay_id) AS n_icustays
        FROM icustays
        GROUP BY subject_id) AS sub_q
    WHERE n_icustays > 1;"""

    icupat = exec_query(q_icupat)  # query output
    df_icupat = pd.DataFrame(icupat, columns=['subjectid', 'n_icustays'])

    # filter for ICU patients with readmissions
    filter_preadm = list(df_icupat.subjectid)
    df_trav = df_trav[df_trav.subjectid.isin(filter_preadm)]

    # Count of Traversal pairs
    icuid = list(df_trav.icustayid.value_counts().index)  # unique subject_id

    main_d = dict()
    for stay in icuid:
        pair_d = dict(Counter(df_trav[df_trav.icustayid == stay].trans))
        pair_d['icustayid'] = stay  # add subjectid key
        main_d[stay] = pair_d

    df_toppairs = df_trav.trans.value_counts(ascending=False).to_frame()
    # df_top = df_toppairs.transpose().iloc[:, 0:11]

    df_pairct = pd.DataFrame.from_dict(main_d, orient='index')

    # drop non-top trans pair cols
    pairs_drop = list(df_toppairs.iloc[10:].index)
    df_pairct.drop(pairs_drop, axis=1, inplace=True)
    return (df_trav, df_pairct)


def feat_pair_trans(DataFrame):
    """
    Function takes a DataFrame and transforms the ICU ward traversal pairs by multiplying their count/frequency by that
    combination's overall probability (apply weight). Thus, extracting a risk score for the top 10 pair of traversals as
    features.

    :param DataFrame: DataFrame
    :return: DataFrame + top 10 traversal/transfer pairs risk score features
    """
    df_trav, df_pairct = trav_pairs()
    df_travpairs = pd.merge(DataFrame, df_pairct, on='icustayid', how='left')


    # Probability Transformation (weight)
    prev_cu = df_trav.prev_cu
    curr_cu = df_trav.curr_cu
    pair_prob = pd.crosstab(prev_cu, curr_cu) / pd.crosstab(prev_cu, curr_cu).sum() # cross-tabulation containing pair probs

    df_pairprob = pair_prob.unstack().to_frame(name='prob').reset_index()
    df_pairprob['trans'] = df_pairprob.prev_cu + '-' + df_pairprob.curr_cu # traversal pair strings
    df_pairprob.drop(['curr_cu', 'prev_cu'], axis=1, inplace=True)
    df_pairprob.set_index('trans', drop=True, inplace=True)

    pairs = ['nonicu-MICU', 'nonicu-SICU', 'nonicu-TSICU', 'nonicu-CSRU',
             'MICU-MICU', 'TSICU-TSICU', 'nonicu-CCU', 'CCU-CCU', 'CSRU-CSRU',
             'SICU-SICU']
    for elem in pairs:
        df_travpairs[elem].fillna(0, inplace=True)
        df_travpairs[elem] = np.round(df_travpairs[elem] * df_pairprob.loc[elem].values[0], 3)

    return df_travpairs


def y_iculos(DataFrame):
    """
    Function extracts the response/dependent variable of continuous data type.

    :param DataFrame: pandas DataFrame containing features
    :return: pandas DataFrame + response
    """
    q_iculos = """SELECT icustay_id, los
    FROM icustays;"""

    iculos = exec_query(q_iculos) # query output
    df_iculos = pd.DataFrame(iculos, columns=['icustayid', 'icu_los'])

    df_final = pd.merge(DataFrame, df_iculos, on='icustayid', how='left')
    df_final.dropna(inplace=True)
    return df_final


def composite_data(interval):
    """

    :param interval:
    :return:
    """
    # Features
    df_icureadm = feat_icureadm()
    df_icureadm_ex = exclusion(df_icureadm) # exclusion criteria
    df_interval = df_icureadm_ex[df_icureadm_ex['readm_days'] <= interval] # interval cut-off (days since last readmission)
    df_previculos = feat_prev_iculos(df_interval) # previous ICU LOS
    df_intratrans = feat_intra_trans(df_previculos) # intra-ICU ward transfers
    df_currcu = feat_curr_careunit(df_intratrans) # current care unit features
    df_dischcu = feat_disch_careunit(df_currcu) # discharge care unit features
    df_staytime = feat_stay_time(df_dischcu) # ICU time (is_day)
    df_intra_risk = feat_riskscore_intraint(df_staytime) # intra-period risk scores
    df_travpairs = feat_pair_trans(df_intra_risk) # traversal pair risk scores

    # Response
    final_df = y_iculos(df_travpairs)

    # Drop unnecessary features used for data engineering/feature extraction
    final_df.drop(['icu_intime', 'icu_outtime', 'prev_icustayid'], axis=1, inplace=1)
    return final_df


if __name__ == "__main__":
    pass
    # print composite_data(interval=30).describe()
    # data = composite_data(30)

