#! usr/bin/bash
#Makes a directory for the event the script is based on
mkdir /data/wiay/undergrads/project781/GW150914/GapPrior/

#Copies over the python script, submission file, and prior to the file
cp ./GW150914_GapPrior.sub /data/wiay/undergrads/project781/GW150914/GapPrior/

cp ./GW150914_GapPrior.py /data/wiay/undergrads/project781/GW150914/GapPrior/

cp ./GapPrior.prior /data/wiay/undergrads/project781/GW150914/GapPrior/

#Opens to the new directory for the event
cd /data/wiay/undergrads/project781/GW150914/GapPrior/

#Makes all the output files in the directory
mkdir results
mkdir error
mkdir out
mkdir log

#runs the condor submission files and checks it status
condor_submit GW150914_GapPrior.sub

condor_q 2324304m