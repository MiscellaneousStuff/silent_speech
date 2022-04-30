"""
Separate file to save mel_spectrograms for DS2 Model Training / Eval
"""

from transduction_model import test, Model
from read_emg import EMGDataset

from data_utils import phoneme_inventory

import torch
from torch import nn

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_path', None, 'Path to model checkpoint')
flags.mark_flag_as_required("checkpoint_path")

def main(unused_argv):
    trainset = EMGDataset(dev=False,test=False)
    devset   = EMGDataset(dev=True)
    testset  = EMGDataset(test=True)

    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    checkpoint_path = FLAGS.checkpoint_path
    state_dict = torch.load(checkpoint_path)
    n_sess = 1 if FLAGS.no_session_embed else state_dict["session_emb.weight"].size(0)
    
    n_phones = len(phoneme_inventory)
    model = Model(num_ins=testset.num_features,
                  num_outs=testset.num_speech_features,
                  num_aux_outs=n_phones,
                  num_recon_outs=8,
                  num_sessions=n_sess,
                  reconstruction_loss=(True if FLAGS.recon_loss_weight > 0.0 else False)).to(device)
    model.load_state_dict(state_dict)

    model.eval()
    datasets = [trainset, devset, testset]
    for dataset in datasets:
        for i, datapoint in enumerate(dataset):
            with torch.no_grad():
                sess = torch.tensor(datapoint['session_ids'], device=device).unsqueeze(0)
                X = torch.tensor(datapoint['emg'], device=device).unsqueeze(0)
                X_raw = torch.tensor(datapoint['raw_emg'], device=device).unsqueeze(0)
                silent = datapoint['silent']
                
                with torch.autocast(
                    enabled=FLAGS.amp,
                    dtype=torch.bfloat16,
                    device_type=device):

                    pred, phoneme_pred, X_recon = model(X, X_raw, sess)
                    pred = pred.squeeze(0)

                    sentence_idx = datapoint["book_location"][1]

                    pred = pred.cpu()
                    pred = pred.float().detach()

                if silent:
                    torch.save(pred, f"./pred_audio/open_vocab_parallel/silent/{sentence_idx}")
                    print("PARALLEL SILENT", datapoint["book_location"], pred.dtype)
                else:
                    if datapoint["book_location"][0] == "books/War_of_the_Worlds.txt":
                        torch.save(pred, f"./pred_audio/open_vocab_parallel/voiced/{sentence_idx}")
                        print("PARALLEL VOICED", datapoint["book_location"], pred.dtype)
                    else:
                        torch.save(pred, f"./pred_audio/open_vocab_non_parallel/voiced/{sentence_idx}")
                        print("NON_PARALLEL VOICED", datapoint["book_location"], pred.dtype)

if __name__ == '__main__':
    app.run(main)