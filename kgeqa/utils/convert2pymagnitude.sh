#!/bin/bash

python -m pymagnitude.converter -i "/Users/Aziz/Dropbox/thesis/code/kge_qa/helpers/resource/wn_names.vec" -o "/Users/Aziz/Dropbox/thesis/code/kge_qa/helpers/resource/wn_names.vec.magnitude"
python -m pymagnitude.converter -i ENT.vec -o ENT.vec.magnitude
