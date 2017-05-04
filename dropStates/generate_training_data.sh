 #!/bin/bash
 PATH_TO_AMUN="/home/dheart/uni_stuff/phd_2/sona_project/amunmt_dropstates/build/bin/amun"
 SCIPTS_DIR="/home/dheart/uni_stuff/phd_2/sona_project/amunmt_dropstates/dropStates"
 PARENT_DIR="/mnt/Storage/sona_exp/de-en_test/dev_state"
 CORPORA_DIR="/mnt/Storage/sona_exp/de-en_test/dev_state/corpora"
 MODEL_DIR="/mnt/Storage/sona_exp/de-en_test/dev_state"
 CORPUS_PREFIX="dev.bpe"
 SOURCE_LANG="de"
 TARGET_LANG="en"
 FILE_SUFFIX=$1
 TARGET_FILE=$CORPORA_DIR/$CORPUS_PREFIX"."$TARGET_LANG"."$FILE_SUFFIX
 SOURCE_FILE=$CORPORA_DIR/$CORPUS_PREFIX"."$SOURCE_LANG"."$FILE_SUFFIX
 read NUM_LINES FILENAME <<< $(wc -l $SOURCE_FILE) || return 1
 
 cd $PARENT_DIR || return 1
 mkdir -p 'states'$FILE_SUFFIX || return 1
 cd 'states'$FILE_SUFFIX || return 1
 mkdir -p dropStates || return 1
 #Decode the sentences
 $PATH_TO_AMUN -c $MODEL_DIR/config.yml --n-best --cpu-threads 4 < $CORPORA_DIR/$CORPUS_PREFIX"."$SOURCE_LANG"."$FILE_SUFFIX > n_best_list || return 1
 #Evaluate BLEU
 $SCIPTS_DIR/model2metric.py n_best_list dropStates $CORPORA_DIR/$CORPUS_PREFIX"."$TARGET_LANG"."$FILE_SUFFIX || return 1
 #Merge training
 cp $SCIPTS_DIR/get_states_with_scores.py dropStates || return 1
 cd dropStates || return 1
 chmod +x get_states_with_scores.py || return 1
 python3 get_states_with_scores.py $NUM_LINES > "../training_data_"$FILE_SUFFIX || return 1
 #Archive
 cd .. || return 1 
 bzip2 -f "training_data_"$FILE_SUFFIX || return 1
 #Clean up
 rm -rf dropStates n_best_list || return 1
