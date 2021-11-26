import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import *
from transformers import BartConfig
from transformers.modeling_outputs import *


class SynSemBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        config.vocab_size = 20  # POSvocab.json
        self.pos_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx)
        self.pos_encoder = BartEncoder(config, self.pos_embed)  # no shared embedding

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_pos_encoder(self):
        return self.pos_encoder

    def get_decoder(self):
        return self.decoder

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

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Semantic Encoder
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Syntactic Encoder
        # pos_encoder_outputs[0] 그대로 쓸지, pooling해서 repeat할지 고민
        if pos_encoder_outputs is None:
            pos_encoder_outputs = self.pos_encoder(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                head_mask=pos_head_mask,
                inputs_embeds=pos_inputs_embeds,
                output_attentions=pos_output_attentions,
                output_hidden_states=pos_output_hidden_states,
                return_dict=pos_return_dict,
            )

        # input sequence length of current batch
        seq_len = encoder_outputs[0].size()[1]

        # pos_encoder_outputs[0]: (batch_size, pos_len, embedding_dim)
        # pos_encoder_outputs[0][:,0,:]: (batch_size, embedding_dim) -> <s>의 hidden state만
        # pos_encoder_outputs[0][:,0,:].unsqueeze(1): (batch_size, 1, embedding_dim)
        # pos_encoder_outputs[0][:,0,:].unsqueeze(1).repeat(1,seq_len,1): (batch_size, seq_len, embedding_dim)
        pos_encoder_outputs_repeated = pos_encoder_outputs[0][:,0,:].unsqueeze(1).repeat(1,seq_len,1)


        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        if return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )


        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0] + pos_encoder_outputs_repeated,  # Syntactic + Semantic
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
