import torch
from models.enc_visual import FeatureFlat
from models.enc_vl import EncoderVL
from models.encodings import DatasetLearnedEncoding
from models import model_util
from torch import nn
from torch.nn import functional as F

import numpy as np


class SoftDotAttention(nn.Module):
    """
    Soft Dot Attention.

    Ref: https://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

        self.c = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
            nn.ReLU())

    def forward(self, h, context, mask=None):  # context will be weighted and concat with h
        """
        Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax 
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        lang_embeds = torch.cat((weighted_context, h), 1)

        lang_embeds = self.tanh(self.linear_out(lang_embeds))
        return lang_embeds, attn


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
        # weighted_context = self.mlp(self.ln_2(weighted_context + img_feats.mean(-1)))
        weighted_context = self.mlp(self.ln_2(weighted_context + text_cls))

        return weighted_context


class ET(nn.Module):
    def __init__(self, args):
        """
        transformer agent
        """
        super().__init__()
        # encoder and visual embeddings
        self.encoder_vl = EncoderVL(args)
        self.args = args

        # XVIEW
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4), )

        # self.attention_layer_vision = SoftDotAttention(49)
        self.attention_layer_vision = CrossAttentionBlock(q_dim=768, kv_dim=1280, n_head=12, )
        self.direction_embedding = nn.Linear(2, 768)

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

        # final touch
        self.init_weights()

    def forward(self, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # embed language
        emb_lang = inputs["lang"]
        im_feature = inputs["frames"]

        # embed frames: [bs,t,512,49] --> [bs,t,512]
        att_frame_feature = torch.zeros((im_feature.shape[0], 0, 768)).cuda()
        for i in range(im_feature.shape[1]):
            # [bs, 512, 49] --> [bs, 768]
            att_single_frame_feature = self.attention_layer_vision(inputs["lang_cls"], im_feature[:, i, :, :])
            att_frame_feature = torch.concat([att_frame_feature, att_single_frame_feature.unsqueeze(1)], dim=1)  # [bs, t, 768]

        emb_frames = att_frame_feature
        emb_directions = self.direction_embedding(inputs["directions"].view(-1, 2)).view(im_feature.shape[0], -1, 768)  # (batch, embedding_size)

        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl(
            emb_lang,
            emb_frames,
            emb_directions,
            inputs['lenths']
        )
        # use outputs corresponding to last visual frames for prediction only
        encoder_out_visual = encoder_out[:, emb_lang.shape[1] + np.max(inputs['lenths']) - 1]
        encoder_out_direction = encoder_out[:, emb_lang.shape[1] + 2 * np.max(inputs['lenths']) - 1]
        # get the output actions
        decoder_input = encoder_out_visual.reshape(-1, self.args.demb)
        action_decoder_input = encoder_out_direction.reshape(-1, self.args.demb)

        # decoder_input = emb_directions[:,-1].reshape(-1, self.args.demb)
        output = self.decoder_2_action_full(action_decoder_input)
        h_sali = self.fc(decoder_input).view(-1, 1, 8, 8)

        # mask prediction;
        pred_saliency = nn.functional.interpolate(h_sali, size=(224, 224), mode='bilinear', align_corners=False)
        # bbox predicton;
        bbox_logits = self.bbox_pred(decoder_input)

        # # get the output objects
        # emb_object_flat = emb_object.view(-1, self.args.demb)
        # decoder_input = decoder_input + emb_object_flat
        # object_flat = self.dec_object(decoder_input)
        # objects = object_flat.view(*encoder_out_visual.shape[:2], *object_flat.shape[1:])
        # output.update({"action": action, "object": objects})

        # (optionally) get progress monitor predictions
        # if self.args.progress_aux_loss_wt > 0:
        #     progress = torch.sigmoid(self.dec_progress(encoder_out_visual))
        #     output["progress"] = progress
        # if self.args.subgoal_aux_loss_wt > 0:
        #     subgoal = torch.sigmoid(self.dec_subgoal(encoder_out_visual))
        #     output["subgoal"] = subgoal
        return output, pred_saliency, bbox_logits

    def compute_batch_loss(self, model_out, gt_dict):
        """
        loss function for Seq2Seq agent
        """
        losses = dict()

        # action loss
        action_pred = model_out["action"].view(-1, model_out["action"].shape[-1])
        action_gt = gt_dict["action"].view(-1)
        pad_mask = action_gt != self.pad

        # Calculate loss only over future actions
        action_pred_mask = gt_dict["driver_actions_pred_mask"].view(-1)

        action_loss = F.cross_entropy(action_pred, action_gt, reduction="none")
        action_loss *= pad_mask.float()
        if not self.args.compute_train_loss_over_history:
            action_loss *= action_pred_mask.float()
        action_loss = action_loss.mean()
        losses["action"] = action_loss * self.args.action_loss_wt

        # object classes loss
        if len(gt_dict["object"]) > 0:
            object_pred = model_out["object"]
            object_gt = torch.cat(gt_dict["object"], dim=0)

            if self.args.compute_train_loss_over_history:
                interact_idxs = gt_dict["obj_interaction_action"].view(-1).nonzero(as_tuple=False).view(-1)
            else:
                interact_idxs = (
                    (gt_dict["driver_actions_pred_mask"] * gt_dict["obj_interaction_action"])
                    .view(-1)
                    .nonzero(as_tuple=False)
                    .view(-1)
                )
            if interact_idxs.nelement() > 0:
                object_pred = object_pred.view(object_pred.shape[0] * object_pred.shape[1], *object_pred.shape[2:])
                object_loss = model_util.obj_classes_loss(object_pred, object_gt, interact_idxs)
                losses["object"] = object_loss * self.args.object_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            subgoal_pred = model_out["subgoal"].squeeze(2)
            subgoal_gt = gt_dict["subgoals_completed"]
            subgoal_loss = F.mse_loss(subgoal_pred, subgoal_gt, reduction="none")
            subgoal_loss = subgoal_loss.view(-1) * pad_mask.float()
            subgoal_loss = subgoal_loss.mean()
            losses["subgoal_aux"] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.progress_aux_loss_wt > 0:
            progress_pred = model_out["progress"].squeeze(2)
            progress_gt = gt_dict["goal_progress"]
            progress_loss = F.mse_loss(progress_pred, progress_gt, reduction="none")
            progress_loss = progress_loss.view(-1) * pad_mask.float()
            progress_loss = progress_loss.mean()
            losses["progress_aux"] = self.args.progress_aux_loss_wt * progress_loss

        # maximize entropy of the policy if asked
        if self.args.entropy_wt > 0.0:
            policy_entropy = -F.softmax(action_pred, dim=1) * F.log_softmax(action_pred, dim=1)
            policy_entropy = policy_entropy.mean(dim=1)
            policy_entropy *= pad_mask.float()
            losses["entropy"] = -policy_entropy.mean() * self.args.entropy_wt

        return losses

    def init_weights(self, init_range=0.1):
        """
        init embeddings uniformly
        """
        pass

    def compute_metrics(self, model_out, gt_dict, metrics_dict, compute_train_loss_over_history):
        """
        compute exact matching and f1 score for action predictions
        """
        preds = model_util.extract_action_preds(model_out, self.pad, self.vocab_out, lang_only=True)
        stop_token = self.vocab_out.word2index("Stop")
        gt_actions = model_util.tokens_to_lang(gt_dict["action"], self.vocab_out, {self.pad, stop_token})
        model_util.compute_f1_and_exact(metrics_dict, [p["action"] for p in preds], gt_actions, "action")
        model_util.compute_obj_class_precision(
            metrics_dict, gt_dict, model_out["object"], compute_train_loss_over_history
        )

    def compute_loss(self, model_outs, gt_dicts):
        """
        compute the loss function for several batches
        """
        # compute losses for each batch
        losses = {}
        for dataset_key in model_outs.keys():
            losses[dataset_key] = self.compute_batch_loss(model_outs[dataset_key], gt_dicts[dataset_key])
        return losses
