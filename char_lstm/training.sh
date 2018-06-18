#!/usr/bin/env bash

##main experiments
python main.py --data '../datasets/penn/unknown' --char_vs_word='char' --save 'outPennUnkChar' --cuda --epochs 20
python main.py --data '../datasets/Unk_subword/penn_unk_subword' --char_vs_word='subword' --save 'outPennUnkUnkSub' --cuda
python main.py --data '../datasets/penn/unknown' --char_vs_word='word' --save 'outPennUnkWord' --cuda

python main.py --data '../datasets/EN/unknown' --char_vs_word='char' --save 'outENUnkChar' --cuda --epochs 20
python main.py --data '../datasets/Unk_subword/EN_unk_subword' --char_vs_word='subword' --save 'outENUnkUnkSub' --cuda
python main.py --data '../datasets/EN/unknown' --char_vs_word='word' --save 'outENUnkWord' --cuda

python main.py --data '../datasets/TR/unknown' --char_vs_word='char' --save 'outTRUnkChar' --cuda --epochs 20
python main.py --data '../datasets/Unk_subword/TR_unk_subword' --char_vs_word='subword' --save 'outTRUnkUnkSub' --cud
python main.py --data '../datasets/TR/unknown' --char_vs_word='word' --save 'outTRUnkWord' --cuda

python main.py --data '../datasets/GE/unknown' --char_vs_word='char' --save 'outGEUnkChar' --cuda --epochs 20
python main.py --data '../datasets/Unk_subword/GE_unk_subword' --char_vs_word='subword' --save 'outGEUnkUnkSub' --cuda
python main.py --data '../datasets/GE/unknown' --char_vs_word='word' --save 'outGEUnkWord' --cuda

#bigger char model
python main.py --data '../datasets/penn/unknown' --char_vs_word='char' --save 'outPennUnkBigChar' --cuda --epochs 20 --nhid 2000
python main.py --data '../datasets/EN/unknown' --char_vs_word='char' --save 'outENUnkBigChar' --cuda --epochs 20 --nhid 2000
python main.py --data '../datasets/GE/unknown' --char_vs_word='char' --save 'outGEUnkBigChar' --cuda --epochs 20 --nhid 2000
python main.py --data '../datasets/TR/unknown' --char_vs_word='char' --save 'outTRUnkBigChar' --cuda --epochs 20 --nhid 2000

#original text
python main.py --data '../datasets/penn/original' --char_vs_word='char' --save 'outPennOrigChar' --cuda  --epochs 20
python main.py --data '../datasets/penn/sub_original' --char_vs_word='subword' --save 'outPennOrigSub' --cuda
python main.py --data '../datasets/penn/original' --char_vs_word='word' --save 'outPennOrigWord' --cuda

python main.py --data '../datasets/EN/original' --char_vs_word='char' --save 'outENOrigChar' --cuda --epochs 20
python main.py --data '../datasets/EN/sub_original' --char_vs_word='subword' --save 'outENOrigSub' --cuda
python main.py --data '../datasets/EN/original' --char_vs_word='word' --save 'outENOrigWord' --cuda

python main.py --data '../datasets/TR/original' --char_vs_word='char' --save 'outTROrigChar' --cuda --epochs 20
python main.py --data '../datasets/TR/sub_original' --char_vs_word='subword' --save 'outTROrigSub' --cuda
python main.py --data '../datasets/TR/original' --char_vs_word='word' --save 'outTROrigWord' --cuda

python main.py --data '../datasets/GE/original' --char_vs_word='char' --save 'outGEOrigChar' --cuda --epochs 20
python main.py --data '../datasets/GE/sub_original' --char_vs_word='subword' --save 'outGEOrigSub' --cuda
python main.py --data '../datasets/GE/original' --char_vs_word='word' --save 'outGEOrigWord' --cuda

#bigger datasets (see what fits on your graphics card)
##additional german datasets with different numbers of unk tokens
python main.py --data '../datasets/GE-Unk/15000' --char_vs_word='char' --save 'outGEUnk15000Char' --cuda --epochs 20
python main.py --data '../datasets/GE-Unk/15000' --char_vs_word='word' --save 'outGEUnk15000Word' --cuda

##additional turkish datasets with different numbers of unk tokens
python main.py --data '../datasets/TR-Unk/15000' --char_vs_word='char' --save 'outTRUnk15000Char' --cuda --epochs 20
python main.py --data '../datasets/TR-Unk/15000' --char_vs_word='word' --save 'outTRUnk15000Word' --cuda

python main.py --data '../datasets/TR-Unk/20000' --char_vs_word='char' --save 'outTRUnk20000Char' --cuda --epochs 20
python main.py --data '../datasets/TR-Unk/20000' --char_vs_word='word' --save 'outTRUnk20000Word' --cuda

python main.py --data '../datasets/TR-Unk/20000' --char_vs_word='char' --save 'outTRUnk20000Char' --cuda --epochs 20
python main.py --data '../datasets/TR-Unk/20000' --char_vs_word='word' --save 'outTRUnk20000Word' --cuda

python main.py --data '../datasets/TR-Unk/50000' --char_vs_word='char' --save 'outTRUnk50000Char' --cuda --epochs 20
python main.py --data '../datasets/TR-Unk/50000' --char_vs_word='word' --save 'outTRUnk50000Word' --cuda