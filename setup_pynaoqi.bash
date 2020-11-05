PLUGIN_V2_PATH="/home/ines/Documents/ENSTA/3A/Asservissement_visuel/Benito/UE52-VS-IK/"
export PYTHONPATH=${PLUGIN_V2_PATH}/external-software/pynaoqi-python2.7-2.1.4.13-linux64:${PYTHONPATH}
#export LD_LIBRARY_PATH=${PLUGIN_V2_PATH}/external-software/naoqi-sdk-2.1.4.13-linux64/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${PLUGIN_V2_PATH}/external-software/naolibs:${LD_LIBRARY_PATH}
# to check if it is OK , you may execute 
printenv | grep PYTHONPATH
printenv | grep LD_LIBRARY_PATH
