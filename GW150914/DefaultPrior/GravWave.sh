#! usr/bin/bash
#Makes a directory for the event the script is based on
mkdir /data/wiay/undergrads/2324304m/GW150914/

#Copies over the python script, submission file, and prior to the file
cp ./GW150914.sub /data/wiay/undergrads/2324304m/GW150914/

cp ./GW150914.py /data/wiay/undergrads/2324304m/GW150914/

cp ./Default.prior /data/wiay/undergrads/2324304m/GW150914/

#Opens to the new directory for the event
cd /data/wiay/undergrads/2324304m/GW150914/

#Makes all the output files in the directory
mkdir results
mkdir error
mkdir out
mkdir log

#runs the condor submission files and checks it status
condor_submit GW150914.sub

condor_q 2324304m