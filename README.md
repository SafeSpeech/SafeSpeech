## SafeSpeech

We propose a data protection framework named SafeSpeech utilizing noise-leading protection (NLP) to prevent publicly uploaded speeches from unauthorized voice cloning.

### Setup

Tested with Ubuntu 20.04.

#### Build Path

Some folders are needed in this repo. The initial directory for the following operations is the root directory.

1. make a directory of checkpoints:

   ```
   mkdir checkpoints
   cd checkpoints
   mkdir base_models
   mkdir LibriTTS
   mkdir CMU_ARCTIC
   ```

2. make a directory of real speech datasets:

   ```
   cd data
   mkdir real
   cd real
   mkdir LibriTTS
   mkdir CMU_ARCTIC
   ```

3. `mkdir perturbation`

After making these directories, please follow:

- Please install ffmpeg for audio transformation.

- The `bert_vits2` folder we were directly forked from <https://github.com/fishaudio/Bert-VITS2> with some modification of file lists.

- Download the pre-trained checkpoints from <https://huggingface.co/OedoSoldier/Bert-VITS2-2.3/tree/main> and move them to `checkpoints/base_models`

- Download the BERT pre-trained models and move them to `bert_vits2/bert`

  - <https://huggingface.co/hfl/chinese-roberta-wwm-ext-large>
  - <https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm>
  - <https://huggingface.co/microsoft/deberta-v3-large>

  or download automatically by the following commands in the `Dataset.3` part.

- Download the pre-trained WavLM model from <https://huggingface.co/microsoft/wavlm-base-plus> and move them to `bert_vits2/slm/wavlm-base-plus`

### Requirements

- At least one GPU for noise generation.
- Python>=3.9.0
- PyTorch>=2.0.1
- pip install -r requirements.txt

### Dataset

In our paper, we have conducted our experiments on two datasets. LibriTTS: <http://www.openslr.org/60/> and download train-clean-100.tar.gz subset; CMU ARCTIC: <http://festvox.org/cmu_arctic/packed/> You can use your customized voices to achieve protection as following, and we use LibriTTS dataset as an example:

1. Move your dataset to `data/real/your_dataset_name`, such as `data/real/LibriTTS`

2. Make your file lists for each dataset and move them to `bert_vits2/filelists/your_dataset.txt`, such as `bert_vits2/filelists/libritts_train_text.txt` we provided, the file list generation can refer [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) for your customized datasets.

3. Generate BERT features to satisfy the conditions of Bert-VITS2 using `bert_gen.py`, such as 

   ```
   python bert_gen.py -c bert_vits2/configs/libritts_bert_vits2.json -n 1
   ```

### Protection

You can choose the data protection method to generate SafeSpeeched perturbation, such as utilizing the "NLP" method:

```
python protect.py --dataset LibriTTS --protection-mode NLP
```

Basic arguments:

- `--dataset`: which dataset to protect. Default: LibriTTS
- `--model`: the surrogate model. Default: Bert_VITS2
- `--batch-size`: the batch size of training and perturbation generation. Default: 27
- `--gpu`: use which GPU. Default:0
- `--random-seed`: the random seed in experiments. Default: 1234
- `--protection-mode`: the protection mode of the SafeSpeech. Default: NLP
- `--checkpoints-path`: the storing dir of the checkpoints. Default: checkpoints/ 
- `--perturbation-path`: the storing dir of the generated perturbation. Default: 
- `--epsilon`: the perturbation radius boundary. Default:8
- `--perturbation-epochs`: the iteration numbers of the noise. Default: 200

For data protection, we provide lots of methods: ['random_noise', 'AdvPoison', 'SEP', 'PTA', 'NLP', 'PeNLP', 'RobustNLP']. Notably, If you want to set the `protection_mode` as "SEP", it is recommended to run fine-tuning at first to generate intermediate checkpoints.

In this experiment, large GPU memories are needed. We set the batch size as 27 on an A800 GPU with 80GB memory. And we provided our generated protected perturbations in our supplemental file--"paper974_checkpoints.zip" with "NLP" protection mode. You can directly use it without optimization from scratch. After downloading, please move it to the `checkpoints/LibriTTS/` folder.

### Train

1. Train on the unprotected dataset:

```
python train.py --dataset LibriTTS
```

After training, you can get 10 intermediate checkpoints in path `checkpoints/LibriTTS/FT`.

2. Train on the SafeSpeeched dataset:

```
python protected_train.py --dataset LibriTTS --protection-mode NLP --epsilon 8
```

The batch size of training on the SafeSpeeched dataset must be the same as the perturbation generation's.

We also provided the checkpoint in our supplemental file--"paper974_checkpoints.zip" which is trained on the SafeSpeech-protected dataset with "NLP" method, and you can utilize it without training. After downloading, please move it to the `checkpoints/LibriTTS/` folder.

### Evaluation

After training the TTS model, the next step is to evaluate the performance. It has two stages for evaluation: synthetic audio generation and metrics evaluation.

1. Synthetic Audio Generation

   ```
   python evaluation/gen_dataset.py --dataset LibriTTS --mode NLP
   ```

   or

   ```
   python evaluation/gen_dataset.py --dataset LibriTTS --mode FT
   ```

   "FT" represents the model state when training on the clean dataset. In this stage, a txt file will be generated to index the audio which is stored in `evaluation/evallists` folder.

2. Metrics Evaluation

   ```
   python evaluation/evaluate.py --eval-list evaluation/evallists/Bert_VITS2_NLP_LibriTTS_text.txt
   ```

   After running this, you can get the three values of metrics utilized in our paper.

In the metrics evaluation process, if the pre-trained file of ECAPA-TDNN cannot be downloaded automatically, you can download from <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb>, and move them to `encoder/` folder.

Last but not least, you can change the `segment_size` in the config file `bert_vits2/configs/your_dataset_config` and some other settings to reproduce the experiments conducted in our paper.

In this repo, we provide Bert-VITS2 as the surrogate model as an example, our proposed method is universal and can serve generative TTS models. Moreover, you can run the evaluation of the effectiveness of our method utilizing our provided checkpoint.
