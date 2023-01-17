#! usr/bin/bash
#Makes a directory for the event the script is based on
mkdir /data/wiay/undergrads/project781/GW150914/PowerLaw/

#Copies over the python script, submission file, and prior to the file
cp ./GW150914_PowerLaw.sub /data/wiay/undergrads/project781/GW150914/PowerLaw/

cp ./GW150914_PowerLaw.py /data/wiay/undergrads/project781/GW150914/PowerLaw/

cp ./PowerLaw.prior /data/wiay/undergrads/project781/GW150914/PowerLaw/

#Opens to the new directory for the event
cd /data/wiay/undergrads/project781/GW150914/PowerLaw/

#Makes all the output files in the directory
mkdir results
mkdir error
mkdir out
mkdir log

#runs the condor submission files and checks it status
condor_submit GW150914_PowerLaw.sub

condor_q 2324304m