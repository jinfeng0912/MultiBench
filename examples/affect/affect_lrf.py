import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


from private_test_scripts.all_in_one import all_in_one_train # noqa
from training_structures.Supervised_Learning import train, test # noqa
from unimodals.common_models import GRUWithLinear, MLP # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat, LowRankTensorFusion # noqa
from torch.nn.utils.rnn import pack_padded_sequence

# Local wrapper to fix lengths tensor issue
class FixedGRUWithLinear(GRUWithLinear):
    def forward(self, x):
        if self.has_padding:
            seq, lengths = x[0], x[1]
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.as_tensor(lengths, dtype=torch.long)
            else:
                lengths = lengths.detach().cpu().long().view(-1)
            x = pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
            hidden = self.gru(x)[1][-1]
        else:
            hidden = self.gru(x)[0]
        if self.dropout:
            hidden = self.dropout_layer(hidden)
        out = self.linear(hidden)
        if self.flatten:
            out = torch.flatten(out, 1)
        if self.output_each_layer:
            return [0, torch.flatten(x, 1), torch.flatten(hidden, 1), self.lklu(out)]
        return out


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = \
    get_dataloader('/mnt/e/Laboratory/datasets/UR_Funny_humor/humor.pkl',
                   task='regression', robust_test=False, max_pad=True, data_type='humor')

# mosi/mosei
# encoders = [GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).cuda(),
#             GRUWithLinear(74, 128, 32, dropout=True, has_padding=True).cuda(),
#             GRUWithLinear(300, 512, 128, dropout=True, has_padding=True).cuda()]
# head = MLP(128, 512, 1).cuda()

# humor/sarcasm
encoders=[FixedGRUWithLinear(371,512,32,dropout=True,has_padding=False, batch_first=True).cuda(), \
    FixedGRUWithLinear(81,256,32,dropout=True,has_padding=False, batch_first=True).cuda(),\
    FixedGRUWithLinear(300,600,128,dropout=True,has_padding=False, batch_first=True).cuda()]
head=MLP(128,512,1).cuda()

# max_seq_len defaults to 50 in get_dataloader; flattened dims = 50 * out_dim per modality
fusion = LowRankTensorFusion([50*32, 50*32, 50*128], 128, 32, flatten=True).cuda()

train(encoders, fusion, head, traindata, validdata, 8, task="regression", optimtype=torch.optim.AdamW,
      early_stop=True, is_packed=False, lr=1e-3, save='humor_lrf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('humor_lrf_best.pt').cuda()

test(model=model, test_dataloaders_all=test_robust, dataset='humor', is_packed=False,
     criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)