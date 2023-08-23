import torch
from models.enc_vl import EncoderVL
from torch import nn

import numpy as np


class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, n_head: int):
        super().__init__()

        self.ln_q_1 = nn.LayerNorm(q_dim)
        self.ln_kv_1 = nn.LayerNorm(kv_dim)
        self.self_attn = nn.MultiheadAttention(q_dim, n_head, kdim=kv_dim, vdim=kv_dim, batch_first=True, dropout=0.1)

        self.ln_2 = nn.LayerNorm(q_dim)
        self.mlp = nn.Sequential(
            nn.Linear(q_dim, q_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(q_dim * 2, q_dim), )

    def forward(self, text_cls, img_feats):
        """
        text_cls: [bs,dim]
        img_feats: [bs,dim,length]
        """
        # reshape
        img_feats = img_feats.permute(0, 2, 1)  # [bs, 512, 49] --> [bs, 49, 512]

        # cross attention
        weighted_context, _ = self.self_attn(query=self.ln_q_1(text_cls.unsqueeze(1)), key=self.ln_kv_1(img_feats), value=self.ln_kv_1(img_feats))  # [bs,1,dim]
        weighted_context = weighted_context.squeeze(1)  # [bs,dim]

        # ffn
        weighted_context = self.mlp(self.ln_2(weighted_context + text_cls))

        return weighted_context


class ET(nn.Module):
    def __init__(self, args):
        """
        transformer agent
        """
        super().__init__()
        self.args = args

        # multi-modal fusion;
        self.encoder_vl = EncoderVL(args)

        # action prediction;
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4), )

        # vision feature aggregate， action embeddings;
        self.attention_layer_vision = CrossAttentionBlock(q_dim=768, kv_dim=1280, n_head=12, )
        self.direction_embedding = nn.Linear(2, 768)

        # mask prediction;
        self.fc = nn.Sequential(nn.Linear(768, 64), nn.Dropout(0.2), nn.ReLU())

        # bbox prediction; 1+4 for labels and bbox;
        self.bbox_pred = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5), )

        self.sprel_linear = nn.Linear(1, 1)

        # final touch
        self.init_weights()

    def forward(self, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # [bs,l,dim]
        emb_lang = inputs["lang"]
        # [ [bs,768] [bs,768],... ]
        im_features = inputs["frames"]
        # [ [bs,768] [bs,768],... ]
        direct_features = inputs["directions"]
        # [[bs,2], [bs,2],...]
        xy_pos = inputs["pos"]

        emb_frames = torch.stack(im_features, dim=1)
        emb_directions = torch.stack(direct_features, dim=1)

        # -------------------------------------------------------------------------------------- #
        # fuse the frames(node) using graph aware attention;
        # -------------------------------------------------------------------------------------- #
        # project the pair-distance matrix, [bs,l,l] -> [bs,l,l,1] -> [bs,1,l,l];
        gmap_pair_dists = self.compute_pair_dist(xy_pos, inputs['lenths'])
        graph_sprels = self.sprel_linear(gmap_pair_dists.unsqueeze(3)).squeeze(3)

        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl(
            emb_lang,
            emb_frames,
            graph_sprels,
            emb_directions,
            inputs['lenths'])

        # use outputs corresponding to last visual frames for prediction only
        encoder_out_visual = encoder_out[:, emb_lang.shape[1] + np.max(inputs['lenths']) - 1]
        encoder_out_direction = encoder_out[:, emb_lang.shape[1] + 2 * np.max(inputs['lenths']) - 1]
        # get the output actions
        decoder_input = encoder_out_visual.reshape(-1, self.args.demb)
        action_decoder_input = encoder_out_direction.reshape(-1, self.args.demb)

        # decoder_input = emb_directions[:,-1].reshape(-1, self.args.demb)
        output = self.decoder_2_action_full(action_decoder_input)
        h_sali = self.fc(decoder_input).view(-1, 1, 8, 8)
        pred_saliency = nn.functional.interpolate(h_sali, size=(224, 224), mode='bilinear', align_corners=False)
        # bbox predicton;
        bbox_logits = self.bbox_pred(decoder_input)

        return output, pred_saliency, bbox_logits

    def embed_frames(self, frames_pad, lang_cls, t):
        """
        embedding the vision feature at time t, also add time embeddings;
        :param lang_cls: global language feature;
        :param frames_pad: image feature at t, [bs,512,49]
        :param t: current time, [1]
        :return: vision embeddings
        """
        # [bs,512,49] -> [bs,768]
        att_single_frame_feature = self.attention_layer_vision(lang_cls, frames_pad)

        embeddings = att_single_frame_feature
        return embeddings

    def embed_actions(self, actions, t):
        """
        embedding the angle at t, also add time embeddings;
        :param actions: current angle encodings,
        :param t: current time, [bs]
        :return: action embeddings
        """
        # [bs,2] -> [bs,768]
        emb_directions = self.direction_embedding(actions)

        embeddings = emb_directions
        return embeddings

    @staticmethod
    def compute_pair_dist(xy_pos, length):
        """
        obtain the graph pair distance for each scence in batch, from the xy postion of each node;
        :param length: trajectory length for each scene;
        :param xy_pos: a list of node xy position, [ t=1,[bs,2], t=2,[bs,2],  ];
        :return: the padding pair dist matrix;
        """
        # [[bs,2], [bs,2],...] -> [bs,t,2]
        xy_pos = np.stack(xy_pos, axis=1)
        max_gmap_len = np.max(length)
        gmap_pair_dists = np.zeros([len(xy_pos), max_gmap_len, max_gmap_len], dtype=np.float32)

        # -------------------------------------------------------------------------------------- #
        # way 1, only aware the nodes before ended;
        # -------------------------------------------------------------------------------------- #
        # for n in range(len(xy_pos)):
        #     xy_nodes, length_nodes = xy_pos[n], length[n]
        #
        #     # for each scene, compute the distance matrix
        #     map_pair_dists = np.zeros([length_nodes, length_nodes], dtype=np.float32)
        #     for i in range(1, length_nodes):
        #         for j in range(i + 1, length_nodes):
        #             map_pair_dists[i, j] = map_pair_dists[j, i] = np.linalg.norm([xy_nodes[i], xy_nodes[j]])
        #
        #     # record and padding;
        #     gmap_pair_dists[n, :length_nodes, :length_nodes] = map_pair_dists

        # -------------------------------------------------------------------------------------- #
        # way 2, aware all nodes including ended nodes;
        # -------------------------------------------------------------------------------------- #
        for n in range(len(xy_pos)):
            xy_nodes = xy_pos[n]

            # for each scene, compute the distance matrix
            map_pair_dists = np.zeros([max_gmap_len, max_gmap_len], dtype=np.float32)
            for i in range(1, max_gmap_len):
                for j in range(i + 1, max_gmap_len):
                    map_pair_dists[i, j] = map_pair_dists[j, i] = np.linalg.norm([xy_nodes[i], xy_nodes[j]])

            # record and padding;
            gmap_pair_dists[n, :max_gmap_len, :max_gmap_len] = map_pair_dists

        gmap_pair_dists = torch.from_numpy(gmap_pair_dists).cuda()
        return gmap_pair_dists

    def init_weights(self, init_range=0.1):
        """
        init embeddings uniformly
        """
        pass


def gen_seq_masks(seq_lens, max_len=None):
    if max_len is None:
        max_len = max(seq_lens)

    batch_size = len(seq_lens)

    masks = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1).cuda()
    masks = masks < seq_lens.unsqueeze(1)
    return masks
