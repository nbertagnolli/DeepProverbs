# Deep Proverbs

This repository holds the code for creating a simple twitter bot using GPT2.  It utilizes tensorflow 1.15 so a docker container is provided for easy usage.
This project originally used a character level RNN and code for that can be found in src/legacy.  All current code is located in deepproverbs.py.  A complete
tutorial is available here if you like supporting my work and buying me a beer, and here if you don't have a medium subscription or dislike buying me beer :'(.

# Setting up the Environment

The associated Makefile and Dockerfile should make playing with the code simple.  To start:

1. build the docker container from the base directory with `make build`

2. run either a jupyter or terminal environment with `make run-jupyter` or `make run`

That's all you need to start playing around with the code.  If you've trained a model and want to
tweet something you can run `make CHECKPOINT_DIR="deepproverbs" TWITTER_CREDS="twitter.json" tweet`

