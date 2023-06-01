From [here](https://github.com/weaviate/t2v-transformers-models), with added support for Apple Silicon (MPS). I rebased and fixed the existing MPS branch to work with the latest version of the module. Inference is at least one order of magnitude faster with MPS.

To run locally:

### Install dependencies and serve module:
```
conda env create --name weaviate
conda activate weaviate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8082
```
The module itself can't be dockerized, because MPS is not available inside the VM that is used by Docker for Mac. If you run the main Weaviate instance inside Docker, you have to pass the module address to the Weviate instance like this: `http://host.docker.internal:8082`.

### Example docker-compose.yml to run weviate with this module:
```
version: '3.9'
services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8080'
    - --scheme
    - http
    image: semitechnologies/weaviate:1.19.5
    ports: 
      - 8080:8080
    restart: on-failure:0
    volumes:
      - ./storage/:/var/lib/weaviate
      - ./backup/:/var/lib/backup
    environment:
      TRANSFORMERS_INFERENCE_API: 'http://host.docker.internal:8082'
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      ENABLE_MODULES: 'backup-filesystem,text2vec-transformers'
      CLUSTER_HOSTNAME: 'node1'
      BACKUP_FILESYSTEM_PATH: "/var/lib/backup/"
    ...
 ```

# transformers inference (for Weaviate)

This is the the inference container which is used by the Weaviate
`text2vec-transformers` module. You can download it directly from Dockerhub
using one of the pre-built images or built your own (as outlined below).

It is built in a way to support any PyTorch or Tensorflow transformers model,
either from the Huggingface Model Hub or from your disk.

This makes this an easy way to deploy your Weaviate-optimized transformers
NLP inference model to production using Docker or Kubernetes.

## Documentation

Documentation for this module can be found [here](https://weaviate.io/developers/weaviate/current/retriever-vectorizer-modules/text2vec-transformers.html).

## Choose your model

### Pre-built images

You can download a selection of pre-built images directly from Dockerhub. We
have chosen publically available models that in our opinion are well suited for
semantic search. 

The pre-built models include:

|Model Name|Image Name|
|---|---|
|`distilbert-base-uncased` ([Info](https://huggingface.co/distilbert-base-uncased))|`semitechnologies/transformers-inference:distilbert-base-uncased`|
|`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ([Info](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2))|`semitechnologies/transformers-inference:sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2`|
|`sentence-transformers/multi-qa-MiniLM-L6-cos-v1` ([Info](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1))|`semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1`|
|`sentence-transformers/multi-qa-mpnet-base-cos-v1` ([Info](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1))|`semitechnologies/transformers-inference:sentence-transformers-multi-qa-mpnet-base-cos-v1`|
|`sentence-transformers/all-mpnet-base-v2` ([Info](https://huggingface.co/sentence-transformers/all-mpnet-base-v2))|`semitechnologies/transformers-inference:sentence-transformers-all-mpnet-base-v2`|
|`sentence-transformers/all-MiniLM-L12-v2` ([Info](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2))|`semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L12-v2`|
|`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` ([Info](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2))|`semitechnologies/transformers-inference:sentence-transformers-paraphrase-multilingual-mpnet-base-v2`|
|`sentence-transformers/all-MiniLM-L6-v2` ([Info](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))|`semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2`|
|`sentence-transformers/multi-qa-distilbert-cos-v1` ([Info](https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1))|`semitechnologies/transformers-inference:sentence-transformers-multi-qa-distilbert-cos-v1`|
|`sentence-transformers/gtr-t5-base` ([Info](https://huggingface.co/sentence-transformers/gtr-t5-base))|`semitechnologies/transformers-inference:sentence-transformers-gtr-t5-base`|
|`sentence-transformers/gtr-t5-large` ([Info](https://huggingface.co/sentence-transformers/gtr-t5-large))|`semitechnologies/transformers-inference:sentence-transformers-gtr-t5-large`|
|`google/flan-t5-base` ([Info](https://huggingface.co/google/flan-t5-base))|`semitechnologies/transformers-inference:sentence-transformers-gtr-t5-base`|
|`google/flan-t5-large` ([Info](https://huggingface.co/google/flan-t5-large))|`semitechnologies/transformers-inference:sentence-transformers-gtr-t5-large`|
|DPR Models|
|`facebook/dpr-ctx_encoder-single-nq-base` ([Info](https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base))|`semitechnologies/transformers-inference:facebook-dpr-ctx_encoder-single-nq-base`|
|`facebook/dpr-question_encoder-single-nq-base` ([Info](https://huggingface.co/facebook/dpr-question_encoder-single-nq-base))|`semitechnologies/transformers-inference:facebook-dpr-question_encoder-single-nq-base`|
|`vblagoje/dpr-ctx_encoder-single-lfqa-wiki` ([Info](https://huggingface.co/vblagoje/dpr-ctx_encoder-single-lfqa-wiki))|`semitechnologies/transformers-inference:vblagoje-dpr-ctx_encoder-single-lfqa-wiki`|
|`vblagoje/dpr-question_encoder-single-lfqa-wiki` ([Info](https://huggingface.co/vblagoje/dpr-question_encoder-single-lfqa-wiki))|`semitechnologies/transformers-inference:vblagoje-dpr-question_encoder-single-lfqa-wiki`|


The above image names always point to the latest version of the inference
container including the model. You can also make that explicit by appending
`-latest` to the image name. Additionally, you can pin the version to one of
the existing git tags of this repository. E.g. to pin `distilbert-base-uncased`
to version `1.0.0`, you can use
`semitechnologies/transformers-inference:distilbert-base-uncased-1.0.0`.

Your favorite model is not included? Open a pull-request to include it or build
a custom image as outlined below.

### Custom build with any huggingface model

You can build a docker image which supports any model from the huggingface
model hub with a two-line Dockerfile.

In the following example, we are going to build a custom image for the
`distilroberta-base` model.

Create a new `Dockerfile` (you do not need to clone this repository, any folder
on your machine is fine), we will name it `distilrobert.Dockerfile`. Add the
following lines to it:

```
FROM semitechnologies/transformers-inference:custom
RUN MODEL_NAME=distilroberta-base ./download.py
```

Now you just need to build and tag your Dockerfile, we will tag it as
`distilroberta-inference`:

```
docker build -f distilroberta.Dockerfile -t distilroberta-inference .
```

That's it! You can now push your image to your favorite registry or reference
it locally in your Weaviate `docker-compose.yaml` using the docker tag
`distilroberta-inference`.

### Custom build with a private / local model

You can build a docker image which supports any model which is compatible with
Huggingface's `AutoModel` and `AutoTokenzier`.

In the following example, we are going to build a custom image for a non-public
model which we have locally stored at `./my-model`.

Create a new `Dockerfile` (you do not need to clone this repository, any folder
on your machine is fine), we will name it `my-model.Dockerfile`. Add the
following lines to it:

```
FROM semitechnologies/transformers-inference:custom
COPY ./my-model /app/models/model
```

The above will make sure that your model end ups in the image at
`/app/models/model`. This path is important, so that the application can find the
model.

Now you just need to build and tag your Dockerfile, we will tag it as
`my-model-inference`:

```
docker build -f my-model.Dockerfile -t my-model-inference .
```

That's it! You can now push your image to your favorite registry or reference
it locally in your Weaviate `docker-compose.yaml` using the docker tag
`my-model-inference`.

### M1 Support 

##### Prerequisites 
> NOTE
> We recommand using virtualenv for this procedure

1. Make sure you macOSX version is  at least MacOS 12.3
2. clone this repo
3. run the pytorch nightly install command 
4. `brew install pkgconfig cmake`
```
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

##### Install
To use pytoech M1 support locally clone the repo and run the following command: 

1. run `pip install -r requirements.txt`
2. `uvicorn app:app --host 0.0.0.0 --port 8080`
