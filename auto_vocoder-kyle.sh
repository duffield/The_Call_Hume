# auto start the vocoder

# load env
source .venv/bin/activate

# automatically restart the process if it finishes
while true; do
    # unless there is a file called "cancel"
    if [ -f cancel ]; then
        echo "cancel file found, exiting"
        exit 1
    fi
    # sleep for 1 second
    sleep 1
    # run the process
    python audio_processor.py --device 4 --audio hume #--config example_chain_config.json
done