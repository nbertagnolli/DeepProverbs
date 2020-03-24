# Command Line Options for the tweet command
CHECKPOINT_DIR = "checkpoint" # defaults
TWITTER_CREDS = "twitter.json"

# DOCKER TASKS
# Build the container
build: ## Build the container
	docker build -t deep-proverbs -f Dockerfile .

run: ## Run the container
	docker run --rm -v $(shell pwd):$(shell pwd) -it deep-proverbs:latest

run-jupyter: ## Run a Jupyter notebook at port 8989
	docker run --rm -v $(shell pwd):$(shell pwd) -it -p 8989:8989 deep-proverbs:latest /bin/sh -c 'cd $(shell pwd); jupyter notebook --allow-root --no-browser --port=8989 --ip=0.0.0.0;'

tweet:  ## Post a tweet
	docker run --rm -v $(shell pwd):$(shell pwd) -it deep-proverbs:latest  /bin/sh -c 'cd $(shell pwd); python3 deepproverbs.py post-tweet $(CHECKPOINT_DIR) $(TWITTER_CREDS);'
