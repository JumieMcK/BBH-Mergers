#! usr/bin/bash
#Makes a directory for the event the script is based on
mkdir /data/wiay/undergrads/project781/GW150914/XPwaveform/

#Copies over the python script, submission file, and prior to the file
cp ./GW150914XP.sub /data/wiay/undergrads/project781/GW150914/XPwaveform/

cp ./GW150914XP.py /data/wiay/undergrads/project781/GW150914/XPwaveform/

cp ./default.prior /data/wiay/undergrads/project781/GW150914/XPwaveform/

#Opens to the new directory for the event
cd /data/wiay/undergrads/project781/GW150914/XPwaveform/

#Makes all the output files in the directory
mkdir results
mkdir error
mkdir out
mkdir log

#runs the condor submission files and checks it status
condor_submit GW150914XP.sub

condor_q 2324304m