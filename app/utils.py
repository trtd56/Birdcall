import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet161
import librosa
import numpy as np

def audio_to_spec(audio, sr):
    spec = librosa.power_to_db(
        librosa.feature.melspectrogram(audio, sr=sr, fmin=20, fmax=16000, n_mels=128)
    )
    return spec.astype(np.float32)

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def predict(path, model):
    x, sr = librosa.load(path, mono=True, res_type="kaiser_fast")
    spec_img = mono_to_color(audio_to_spec(x, sr))
    input_x = torch.tensor(spec_img).unsqueeze(0).transpose(2, 3).transpose(1, 2)/255.
    preds = model(input_x)
    pred_i = int(preds[0].argmax())
    pred = INV_BIRD_CODE[pred_i]
    rate = float(preds.sigmoid()[0][pred_i])
    return f"{pred}: {rate}"

INV_BIRD_CODE = {0: 'aldfly',
 1: 'ameavo',
 2: 'amebit',
 3: 'amecro',
 4: 'amegfi',
 5: 'amekes',
 6: 'amepip',
 7: 'amered',
 8: 'amerob',
 9: 'amewig',
 10: 'amewoo',
 11: 'amtspa',
 12: 'annhum',
 13: 'astfly',
 14: 'baisan',
 15: 'baleag',
 16: 'balori',
 17: 'banswa',
 18: 'barswa',
 19: 'bawwar',
 20: 'belkin1',
 21: 'belspa2',
 22: 'bewwre',
 23: 'bkbcuc',
 24: 'bkbmag1',
 25: 'bkbwar',
 26: 'bkcchi',
 27: 'bkchum',
 28: 'bkhgro',
 29: 'bkpwar',
 30: 'bktspa',
 31: 'blkpho',
 32: 'blugrb1',
 33: 'blujay',
 34: 'bnhcow',
 35: 'boboli',
 36: 'bongul',
 37: 'brdowl',
 38: 'brebla',
 39: 'brespa',
 40: 'brncre',
 41: 'brnthr',
 42: 'brthum',
 43: 'brwhaw',
 44: 'btbwar',
 45: 'btnwar',
 46: 'btywar',
 47: 'buffle',
 48: 'buggna',
 49: 'buhvir',
 50: 'bulori',
 51: 'bushti',
 52: 'buwtea',
 53: 'buwwar',
 54: 'cacwre',
 55: 'calgul',
 56: 'calqua',
 57: 'camwar',
 58: 'cangoo',
 59: 'canwar',
 60: 'canwre',
 61: 'carwre',
 62: 'casfin',
 63: 'caster1',
 64: 'casvir',
 65: 'cedwax',
 66: 'chispa',
 67: 'chiswi',
 68: 'chswar',
 69: 'chukar',
 70: 'clanut',
 71: 'cliswa',
 72: 'comgol',
 73: 'comgra',
 74: 'comloo',
 75: 'commer',
 76: 'comnig',
 77: 'comrav',
 78: 'comred',
 79: 'comter',
 80: 'comyel',
 81: 'coohaw',
 82: 'coshum',
 83: 'cowscj1',
 84: 'daejun',
 85: 'doccor',
 86: 'dowwoo',
 87: 'dusfly',
 88: 'eargre',
 89: 'easblu',
 90: 'easkin',
 91: 'easmea',
 92: 'easpho',
 93: 'eastow',
 94: 'eawpew',
 95: 'eucdov',
 96: 'eursta',
 97: 'evegro',
 98: 'fiespa',
 99: 'fiscro',
 100: 'foxspa',
 101: 'gadwal',
 102: 'gcrfin',
 103: 'gnttow',
 104: 'gnwtea',
 105: 'gockin',
 106: 'gocspa',
 107: 'goleag',
 108: 'grbher3',
 109: 'grcfly',
 110: 'greegr',
 111: 'greroa',
 112: 'greyel',
 113: 'grhowl',
 114: 'grnher',
 115: 'grtgra',
 116: 'grycat',
 117: 'gryfly',
 118: 'haiwoo',
 119: 'hamfly',
 120: 'hergul',
 121: 'herthr',
 122: 'hoomer',
 123: 'hoowar',
 124: 'horgre',
 125: 'horlar',
 126: 'houfin',
 127: 'houspa',
 128: 'houwre',
 129: 'indbun',
 130: 'juntit1',
 131: 'killde',
 132: 'labwoo',
 133: 'larspa',
 134: 'lazbun',
 135: 'leabit',
 136: 'leafly',
 137: 'leasan',
 138: 'lecthr',
 139: 'lesgol',
 140: 'lesnig',
 141: 'lesyel',
 142: 'lewwoo',
 143: 'linspa',
 144: 'lobcur',
 145: 'lobdow',
 146: 'logshr',
 147: 'lotduc',
 148: 'louwat',
 149: 'macwar',
 150: 'magwar',
 151: 'mallar3',
 152: 'marwre',
 153: 'merlin',
 154: 'moublu',
 155: 'mouchi',
 156: 'moudov',
 157: 'norcar',
 158: 'norfli',
 159: 'norhar2',
 160: 'normoc',
 161: 'norpar',
 162: 'norpin',
 163: 'norsho',
 164: 'norwat',
 165: 'nrwswa',
 166: 'nutwoo',
 167: 'olsfly',
 168: 'orcwar',
 169: 'osprey',
 170: 'ovenbi1',
 171: 'palwar',
 172: 'pasfly',
 173: 'pecsan',
 174: 'perfal',
 175: 'phaino',
 176: 'pibgre',
 177: 'pilwoo',
 178: 'pingro',
 179: 'pinjay',
 180: 'pinsis',
 181: 'pinwar',
 182: 'plsvir',
 183: 'prawar',
 184: 'purfin',
 185: 'pygnut',
 186: 'rebmer',
 187: 'rebnut',
 188: 'rebsap',
 189: 'rebwoo',
 190: 'redcro',
 191: 'redhea',
 192: 'reevir1',
 193: 'renpha',
 194: 'reshaw',
 195: 'rethaw',
 196: 'rewbla',
 197: 'ribgul',
 198: 'rinduc',
 199: 'robgro',
 200: 'rocpig',
 201: 'rocwre',
 202: 'rthhum',
 203: 'ruckin',
 204: 'rudduc',
 205: 'rufgro',
 206: 'rufhum',
 207: 'rusbla',
 208: 'sagspa1',
 209: 'sagthr',
 210: 'savspa',
 211: 'saypho',
 212: 'scatan',
 213: 'scoori',
 214: 'semplo',
 215: 'semsan',
 216: 'sheowl',
 217: 'shshaw',
 218: 'snobun',
 219: 'snogoo',
 220: 'solsan',
 221: 'sonspa',
 222: 'sora',
 223: 'sposan',
 224: 'spotow',
 225: 'stejay',
 226: 'swahaw',
 227: 'swaspa',
 228: 'swathr',
 229: 'treswa',
 230: 'truswa',
 231: 'tuftit',
 232: 'tunswa',
 233: 'veery',
 234: 'vesspa',
 235: 'vigswa',
 236: 'warvir',
 237: 'wesblu',
 238: 'wesgre',
 239: 'weskin',
 240: 'wesmea',
 241: 'wessan',
 242: 'westan',
 243: 'wewpew',
 244: 'whbnut',
 245: 'whcspa',
 246: 'whfibi',
 247: 'whtspa',
 248: 'whtswi',
 249: 'wilfly',
 250: 'wilsni1',
 251: 'wiltur',
 252: 'winwre3',
 253: 'wlswar',
 254: 'wooduc',
 255: 'wooscj2',
 256: 'woothr',
 257: 'y00475',
 258: 'yebfly',
 259: 'yebsap',
 260: 'yehbla',
 261: 'yelwar',
 262: 'yerwar',
 263: 'yetvir'}

class BirdcallNet(nn.Module):
    def __init__(self):
        super(BirdcallNet, self).__init__()
        densenet = densenet161(pretrained=False)
        self.features = densenet.features

        self.l8_a = nn.Conv1d(2208, 264, 1, bias=False)
        self.l8_b = nn.Conv1d(2208, 264, 1, bias=False)

    def forward(self, x, perm=None, gamma=None):
        # input: (batch, channel, Hz, time)
        frames_num = x.shape[3]
        x = x.transpose(3, 2)  # (batch, channel, time, Hz)
        h = self.features(x)  # (batch, unit, time, Hz)

        h = F.relu(h, inplace=True)
        h  = torch.mean(h, dim=3)  # (batch, unit, time)
 
        xa = self.l8_a(h)  # (batch, n_class, time)
        xb = self.l8_b(h)  # (batch, n_class, time)
        xb = torch.softmax(xb, dim=2)

        pseudo_label = (xa.sigmoid() >= 0.5).float()
        clipwise_preds = torch.sum(xa * xb, dim=2)
        attention_preds = xb

        return clipwise_preds
