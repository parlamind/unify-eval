from setuptools import setup, find_packages

install_requires = [
    "absl-py==0.9.0",
    "astor==0.8.1",
    "astunparse==1.6.3",
    "beautifulsoup4==4.9.1",
    "bleach==3.1.5",
    "blis==0.4.1",
    "boto==2.49.0",
    "boto3==1.14.20",
    "botocore==1.17.20",
    "Bottleneck==1.3.2",
    "bpemb==0.3.0",
    "cachetools==4.1.1",
    "catalogue==2.0.0",
    "certifi==2020.6.20",
    "chardet==3.0.4",
    "cleantext==1.1.3",
    "click==7.1.2",
    "colorama==0.4.3",
    "cycler==0.10.0",
    "cymem==2.0.3",
    "dialogflow==1.0.0",
    "docutils==0.15.2",
    "fastprogress==0.2.3",
    "filelock==3.0.12",
    "future==0.18.2",
    "gast==0.3.3",
    "gensim==3.8.3",
    "google-api-core==1.21.0",
    "google-auth==1.19.0",
    "google-auth-oauthlib==0.4.1",
    "google-pasta==0.2.0",
    "googleapis-common-protos==1.52.0",
    "grpcio==1.30.0",
    "h5py==2.10.0",
    "idna==2.10",
    "imageio==2.9.0",
    "importlib-metadata==1.7.0",
    "jmespath==0.10.0",
    "joblib==0.16.0",
    "Keras==2.4.3",
    "Keras-Applications==1.0.8",
    "Keras-Preprocessing==1.1.2",
    "keyring==21.2.1",
    "kiwisolver==1.2.0",
    "Markdown==3.2.2",
    "matplotlib==3.2.2",
    "murmurhash==1.0.2",
    "nltk==3.5",
    "numexpr==2.7.1",
    "numpy==1.19.0",
    "nvidia-ml-py3==7.352.0",
    "oauthlib==3.1.0",
    "opt-einsum==3.2.1",
    "packaging==20.4",
    "pandas==1.0.5",
    "Pillow==7.2.0",
    "pkginfo==1.5.0.1",
    "plac==1.2.0",
    "plotly==4.8.2",
    "preshed==3.0.2",
    "protobuf==3.12.2",
    "pyaml==20.4.0",
    "pyasn1==0.4.8",
    "pyasn1-modules==0.2.8",
    "Pygments==2.6.1",
    "pyparsing==2.4.7",
    "python-dateutil==2.8.1",
    "pytz==2020.1",
    "PyYAML==5.3.1",
    "readme-renderer==26.0",
    "regex==2020.6.8",
    "requests==2.24.0",
    "requests-oauthlib==1.3.0",
    "requests-toolbelt==0.9.1",
    "retrying==1.3.3",
    "rfc3986==1.4.0",
    "rsa==4.6",
    "s3transfer==0.3.3",
    "sacremoses==0.0.43",
    "scikit-learn==0.23.1",
    "scikit-optimize==0.7.4",
    "scipy==1.4.1",
    "seaborn==0.10.1",
    "sentencepiece==0.1.91",
    "six==1.15.0",
    "smart-open==2.1.0",
    "soupsieve==2.0.1",
    "spacy==2.3.2",
    "srsly==2.2.0",
    "tensorboard==2.2.2",
    "tensorboard-plugin-wit==1.7.0",
    "tensorboardX==2.1",
    "tensorflow==2.2.0",
    "tensorflow-estimator==2.2.0",
    "termcolor==1.1.0",
    "thinc==7.4.1",
    "threadpoolctl==2.1.0",
    "tokenizers==0.8.1.rc2",
    "torch==1.5.1",
    "torch-vision==0.1.6.dev0",
    "torchvision==0.6.1",
    "tqdm==4.47.0",
    "transformers==3.0.2",
    "twine==3.2.0",
    "urllib3==1.25.9",
    "wasabi==0.7.0",
    "webencodings==0.5.1",
    "Werkzeug==1.0.1",
    "wrapt==1.12.1",
    "zipp==3.1.0",

]

setup(
    name='unify-eval',
    packages=find_packages(include=['unify_eval', "unify_eval.*"]),
    description='',
    version='1.0.10',
    url='https://github.com/parlamind/unify-eval',
    author='marlon.betz@parlamind.com, mohamed.balabel@parlamind.com',
    author_email=' marlon.betz@parlamind.com, mohamed.balabel@parlamind.com',
    install_requires=install_requires
)
