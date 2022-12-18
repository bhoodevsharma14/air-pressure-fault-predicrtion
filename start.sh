#!bin/sh
nphup airflow scheduler &
airflow webserver
