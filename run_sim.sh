ORIGINAL_DIR=$(pwd)
cd /home/duy/RPL-TinyML/contiki-ng/tools/cooja/
./gradlew run --args='--no-gui /home/duy/RPL-TinyML/RPL.csc --logdir=/home/duy/RPL-TinyML/simulation_logs'
cd $ORIGINAL_DIR
