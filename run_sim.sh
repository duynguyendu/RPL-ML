ORIGINAL_DIR=$(pwd)
cd ./contiki-ng/tools/cooja/
./gradlew run --args='--no-gui ../../../RPL.csc --logdir=../../../simulation_logs'
cd $ORIGINAL_DIR
