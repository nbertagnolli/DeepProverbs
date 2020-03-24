from typing import List, Optional
import json
import os
import sys

if "linux" in sys.platform:
    os.environ["LC_AL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"
else:
    os.environ["LC_AL"] = "en_US.utf-8"
    os.environ["LANG"] = "en_US.utf-8"

import click
import gpt_2_simple as gpt2
import numpy as np
import tweepy


def generate_text(
    checkpoint_dir: str,
    length: int,
    temperature: float,
    destination_path: Optional[str],
    prefix: Optional[str],
    return_as_list: bool = False,
) -> List[str]:
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, checkpoint_dir=checkpoint_dir)
    text = gpt2.generate(
        sess,
        checkpoint_dir=checkpoint_dir,
        length=length,
        temperature=temperature,
        destination_path=destination_path,
        prefix=prefix,
        return_as_list=return_as_list,
    )
    return text


def tweet(
    checkpoint_dir: str,
    twitter_credential_path: str,
    length: int = 1024,
    temperature: float = 0.8,
    prefix: Optional[str] = None,
    delimiter: str = "\n==========\n",
):
    print(os.getcwd())
    # Parse the credentials for the twitter bot
    with open(twitter_credential_path, "r") as json_file:
        twitter_creds = json.load(json_file)

    # Set the credentials based on the credentials file
    CONSUMER_KEY = twitter_creds["consumer_key"]
    CONSUMER_SECRET = twitter_creds["consumer_secret"]
    ACCESS_KEY = twitter_creds["access_key"]
    ACCESS_SECRET = twitter_creds["access_secret"]

    # Authenticate with the Twitter API
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    api = tweepy.API(auth)

    # Generate some text
    generated_text = generate_text(
        checkpoint_dir, length, temperature, None, prefix, return_as_list=True
    )

    split_text = generated_text[0].split(delimiter)

    # Filter out all examples which are longer than twitter's 280 character limit
    valid_text = [x for x in split_text if len(x) <= 280]

    # TWEET!!!
    current_tweet = np.random.choice(valid_text, 1)
    api.update_status(current_tweet[0])
    print(current_tweet)


@click.group()
def main():
    pass


@main.command("finetune")
@click.option(
    "--model-name", type=str, default="124M", help="Can be 117M, 124M, or 355M"
)
@click.option("--text-path", type=str, default="./data/mtg_combined.txt")
@click.option("--checkpoint-dir", type=str, default="checkpoint")
@click.option("--num-steps", type=int, default=3000)
@click.option("--sample-length", type=int, default=1023)
@click.option("--save-every", type=int, default=None)
def finetune(
    model_name: str,
    text_path: str,
    checkpoint_dir: str,
    num_steps: int,
    sample_length: int,
    save_every: Optional[int],
) -> None:

    # Download the model if it is not present
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)

    sess = gpt2.start_tf_sess()

    if save_every is None:
        save_every = int(num_steps / 4)

    gpt2.finetune(
        sess,
        text_path,
        model_name=model_name,
        steps=num_steps,
        sample_length=sample_length,
        save_every=save_every,
        checkpoint_dir=checkpoint_dir,
    )  # steps is max number of training steps

    gpt2.generate(sess, checkpoint_dir=checkpoint_dir)


@main.command("generate")
@click.option("--checkpoint-dir", type=str, default="checkpoint")
@click.option("--length", type=int, default=1023)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="lower more consistent, higher more fun",
)
@click.option("--destination-path", type=str, default=None)
@click.option("--prefix", type=str, default=None)
def generate(
    checkpoint_dir: str,
    length: int,
    temperature: float,
    destination_path: str,
    prefix: Optional[str],
) -> None:
    generate_text(checkpoint_dir, length, temperature, destination_path, prefix)


@main.command("post-tweet")
@click.argument("checkpoint-dir", type=str)
@click.argument("twitter-credential-path", type=str)
def post_tweet(checkpoint_dir: str, twitter_credential_path: str):
    tweet(checkpoint_dir, twitter_credential_path)


if __name__ == "__main__":
    main()
