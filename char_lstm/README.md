# Word-, subword- and character-level language modeling RNN

The `main.py` script trains the specified model and accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --nhid NHID        number of hidden units per layer
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --char_vs_word     choose whether to train a word-, subword- or character-based model
  --batch-size N     batch size
  --bptt BPTT        sequence length
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path and name to save the final model and logging files
  --resume_training  resume training from files (loading the model specified in save and overwrites it)
```

With these arguments, a variety of models can be tested.
All our trainings are listed in `training.sh`

With `generate.py` a pretrained model can be used to generate new text. It takes the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus that the model was trained on
  --char_vs_word     choose whether to use a word-, subword- or character-based model
  --cuda             use CUDA
  --checkpoint       path to the pretrained model to use
  --outf             path to the txt file to write the result to
  --words            number of atomic units to generate
  --seed             random seed
```
