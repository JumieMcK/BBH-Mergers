#! usr/bin/bash
#Makes a directory for the event the script is based on
mkdir /data/wiay/undergrads/2324304m/ChirpMass/

#Copies over the python script, submission file, and prior to the file
cp ./GW150914_Chirp.sub /data/wiay/undergrads/2324304m/ChirpMass/

cp ./GW150914_Chirp.py /data/wiay/undergrads/2324304m/ChirpMass/

cp ./ChirpMass.prior /data/wiay/undergrads/2324304m/ChirpMass/

cp ./chirpmass_prior_data.txt /data/wiay/undergrads/2324304m/ChirpMass/
#Opens to the new directory for the event
cd /data/wiay/undergrads/2324304m/ChirpMass/

#Makes all the output files in the directory
mkdir results
mkdir error
mkdir out
mkdir log

#runs the condor submission files and checks it status
condor_submit GW150914_Chirp.sub

condor_q 2324304m