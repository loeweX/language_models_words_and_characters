# Word-, subword- and character-level language modeling RNN

Start training the specified model by running:

```
python main.py 
```
You can choose whether to train a word-, subword- or character-based model by setting the `--char_vs_word` option to `word`, `subword` or `char`. 

Calling

```
python generate.py
```
will use a pretrained model to generate new text. Simply provide the model checkpoint as the `--checkpoint` argument. Don't forget to set the `--char_vs_word` option accordingly.
