# KGE QA 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple-**Q**uestions **A**nswering System based on **K**nowledge **G**raphs **E**mbeddings.

#### Setup

```
$ virtualenv .env -p python3.7
$ source .env/bin/activate
$ pip install -r requirements.txt
```

#### Getting Started

Run in command line mode:

```bash
$ python -m kgeqa.main
```

Run in web browser mode - requires `streamlit`:

```bash
$ streamlit run app.py
``` 

#### Building KGE for a new domain knowledge

We can build new KGE for a new KG dataset either from the CLI or UI (streamlit interface)

From command-line:

```bash
$ python -m kgeqa.build_new_model -csv data/sample1_KG.csv
Started a model builder for data from: data/sample1_KG.csv
Building a new embedding model for 15 tokens ..
Done. See output: data/ENT.vec
Building a new embedding model for 7 tokens ..
Done. See output: data/REL.vec
Converting models to .magnitude format ..
Loading vectors... (this may take some time)
Found 15 key(s)
Each vector has 300 dimension(s)
Creating magnitude format...
Writing vectors... (this may take some time)
...
Successfully converted 'data/ENT.vec' to 'data/ENT.vec.magnitude'!
...
Successfully converted 'data/REL.vec' to 'data/REL.vec.magnitude'!
Done.
```


Description of the generated models:

- `data/ENT.vec` Entity model in `.txt` format (intermediate result - not used in the app)
- `data/ENT.vec.magnitude` Entity model in `PyMagnitude` format
- `data/REL.vec` Relation model in `.txt` format (intermediate result - not used in the app)
- `data/REL.vec.magnitude` Relation model in `PyMagnitude` format 


#### Author

- Aziz Altowayan (Nov. 2019)
