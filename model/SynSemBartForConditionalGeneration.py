import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.bart.modeling_bart import *
from transformers import BartConfig
from SynSemBartModel import SynSemBartModel
# from transformers import BartTokenizer
# from scipy import spatial
# from sent2vec.vectorizer import Vectorizer
from typing import Iterable, Optional, Tuple
import numpy as np
from torch.nn import functional as F
from CustomGenerationMixin import CustomGenerationMixin


# class SynSemBartForConditionalGeneration(BartPretrainedModel, CustomGenerationMixin):
class SynSemBartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = SynSemBartModel(config)  # custom BartModel
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        # self.tokenizer.add_tokens('<name>')

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_pos_encoder(self):
        return self.model.get_pos_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

            pos_input_ids=None,
            pos_attention_mask=None,
            pos_head_mask=None,
            pos_inputs_embeds=None,
            pos_output_attentions=None,
            pos_output_hidden_states=None,
            pos_return_dict=None,
            pos_encoder_outputs=None,

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            pos_input_ids=pos_input_ids,
            pos_attention_mask=pos_attention_mask,
            pos_head_mask=pos_head_mask,
            pos_inputs_embeds=pos_inputs_embeds,
            pos_output_attentions=pos_output_attentions,
            pos_output_hidden_states=pos_output_hidden_states,
            pos_return_dict=pos_return_dict,
            pos_encoder_outputs=pos_encoder_outputs,
        )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            """
            decoded_labels = []
            decoded_logits = []
            for i in range(len(labels)):
                decoded_labels.append(self.tokenizer.decode(labels[i], skip_special_tokens=True))
                decoded_logits.append(self.tokenizer.decode(lm_logits[i].argmax(-1), skip_special_tokens=True))

            vectorizer1 = Vectorizer()
            vectorizer1.bert(decoded_labels)
            vectors_labels = vectorizer1.vectors

            vectorizer2 = Vectorizer()
            vectorizer2.bert(decoded_logits)
            vectors_logits = vectorizer2.vectors

            for v_label, v_logit in zip(vectors_labels, vectors_logits):
                masked_lm_loss += spatial.distance.cosine(v_label, v_logit)
            """

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    # override this method
    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            pos_encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "pos_encoder_outputs": pos_encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # override this method
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs):

        if "encoder_outputs" not in model_kwargs:
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("pos_")
            }

            # retrieve encoder hidden states
            encoder = self.get_encoder()

            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)

            pos_encoder = self.get_pos_encoder()

            model_kwargs["pos_encoder_outputs"]: ModelOutput = pos_encoder(
                                                                input_ids=model_kwargs['pos_input_ids'],
                                                                attention_mask = model_kwargs['pos_attention_mask'],
                                                                return_dict=True
                                                            )

        return model_kwargs