#! usr/bin/bash
#Makes a directory for the event the script is based on
mkdir /data/wiay/undergrads/2324304m/GW190521/

#Copies over the python script, submission file, and prior to the file
cp ./GW190521.sub /data/wiay/undergrads/2324304m/GW190521/

cp ./GW190521.py /data/wiay/undergrads/2324304m/GW190521/

cp ./Default.prior /data/wiay/undergrads/2324304m/GW190521/

#Opens to the new directory for the event
cd /data/wiay/undergrads/2324304m/GW190521/

#Makes all the output files in the directory
mkdir results
mkdir error
mkdir out
mkdir log

#runs the condor submission files and checks it status
condor_submit GW190521.sub

condor_q 2324304m
