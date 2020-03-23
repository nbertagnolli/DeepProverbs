# Deep Proverbs

This repository holds the code for training a character level RNN twitter bot.  If you're interested in learning
more about character level models checkout Andrej Karpathy's great
<a href=http://karpathy.github.io/2015/05/21/rnn-effectiveness/>blog post</a> on the subject.  We implement a slightly
different model here based on the architecture in this <a href=https://arxiv.org/pdf/1308.0850.pdf>paper</a>
by Alex Graves.  This model differs from that paper in a few ways, most noteably that only four layers were used instead
of 7 and dropout was applied at each layer. This project relies on TensorFlow, Keras, numpy, tweepy, and enchant so if
you want to run it make sure that you have those libraries installed.  Docker image coming soon : ).

# Using Docker

If you'd like to run the project but don't want to install everything feel free to use the provided Docker file to build
an environment capable of running this project.  It is important to note that this Dockerfile does not support the GPU
so if you want to run big models you should probably not use this Dockerfile.  To build the Docker image from the shell
run:

`docker build -t tensorflow-cpu .`

This will take a little while but when it's finished running to check that the image was created type:

`docker images`

This will list out all of the Docker images that you have on your system make sure tensorflow-cpu is there.  Now to run
the Docker image type:

`docker run -v ~:/usr/home -it tensorflow-cpu`

This will mount your home directory to /usr/home in the container and open an interactive terminal. Now inside of of the
container just navigate to this repository and run the scripts.

# Running the project.
This project takes in raw text files and trains a character level model.  To start I would recommend running my
clean_text.py script to remove unnecessary characters and numbers by putting the following command in the shell:

`python clean_text.py --path /path/to/file/ --files file_name_1 file_name_2`

Now if that worked you should have some new files called file_name_1_clean you can now use these to train a model by
running:

`python train_model.py --file_path /path/to/file/filename --model_name my_model --context_size 100 --n_hidden 650 \
--dropout .4  --epochs 40 --batch_size 128`

This will train a model with a 100 character context window, 650 units in each hidden layer, and 40% dropout for 40
epochs with a batch size of 128.  It will save off the model state every time it finishes an epoch.

Once you've trained a model you can start generating text by running:

`python generate_text.py --model_path /path/to/model/file --model_spec /path/to/model/spec --n_chars 100`

Enjoy!
