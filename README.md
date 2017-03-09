# MIMIC-III ICU Readmission

## Project Description
### Purpose
Create a predictive model to identify patients proned to ICU readmissions.

### Data Source: MIMIC-III
MIMIC-III (**M**edical **I**nformation **M**art for **I**ntensive **C**are III) is a large and 
publicly-available database. It's comprised of de-identified health-related data associated with
over 40,000 patients that have stayed in the critical care units of the Beth Israel Deaconess
Medical Center (2001-2012).

The database includes information such as demographics, vitals, laboratory tests, procedures,
medications, clinician notes, and so on. For more information please visit the [documentation].  
[documentation]: https://mimic.physionet.org/about/mimic/

### Database Deployment
The project implements a PostgreSQL database that is ran on a virtual machine (VM). The VM handles
all provisioning, including installation and database import.

The Postgres database can be accessed remotely on Python leveraging the `psycopg2` Python package.
Likewise, it can be accessed through Postgres' psql terminal by specifying the localhost and port of
the VM database:  
`psql -h localhost -p 2345 mimic mimic`

For more information or installation of the VM locally, please visit the project repo: [MIMIC-III VM]  
[MIMIC-III VM]: https://github.com/nsh87/mimic-iii-vm

## Project Structure
### IPython dir
The IPython directory contains drafts of efforts of feature engineering to arrive at a composite dataset
for predictive modeling.

### src dir
The `src` directory is the main project's files. It contains Python files (`.py`) related to the
design of the predictive model.

####  `db.py`
The `db.py` file's primary purpose is to output a composite DataFrame to be used for predictve modeling.  
It encompasses database connections, query execution, and feature engineering.

#### `clf.py`
The main purpose of the `clf.py` file is to run the predictive model and generate a fitted model and its corresponding
results (e.g. metrics, visualizations).