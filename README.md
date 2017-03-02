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

For more information or installation of the VM locally, please visit the project repo: [MIMIC-III VM]  
[MIMIC-III VM]: https://github.com/nsh87/mimic-iii-vm

### Project Structure
#### IPython dir
The IPython directory contains drafts of efforts of feature engineering to arrive at a composite dataset
for predictive modeling.

