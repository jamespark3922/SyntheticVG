#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..osprey_arch import OspreyMetaModel, OspreyMetaForCausalLM

from ..layer import MaskExtractor

class OspreyMistralConfig(MistralConfig):
    model_type = "osprey_mistral"


class OspreyMistralModel(OspreyMetaModel, MistralModel):
    config_class = OspreyMistralConfig

    def __init__(self, config: MistralConfig):
        super(OspreyMistralModel, self).__init__(config)


class OspreyMistralForCausalLM(MistralForCausalLM, OspreyMetaForCausalLM):
    config_class = OspreyMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)

        config.rope_scaling = None
        self.model = OspreyMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.mask_extractor = MaskExtractor(out_dim=config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        img_metas = None,
        masks = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        max_length = self.config.max_position_embeddings
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, masks, attention_mask, past_key_values, labels, images)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.bfloat16()
  
        self.model = self.model.bfloat16()

        # Truncate inputs to the maximum length
        if inputs_embeds is not None and inputs_embeds.size(1) > max_length:
            inputs_embeds = inputs_embeds[:, :max_length, :]
        if attention_mask is not None and attention_mask.size(1) > max_length:
            attention_mask = attention_mask[:, :max_length]
        if labels is not None and labels.size(1) > max_length:
            labels = labels[:, :max_length]
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        masks = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        past_key_values = kwargs.pop("past_key_values", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            input_ids, attention_mask, past_key_values, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(inputs, masks, attention_mask, past_key_values, None, images)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    # def prepare_inputs_for_generation(
    #     self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    # ):
    #     if past_key_values:
    #         input_ids = input_ids[:, -1:]

    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and past_key_values is None:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         model_inputs = {"input_ids": input_ids}

    #     model_inputs.update(
    #         {
    #             "past_key_values": past_key_values,
    #             "use_cache": kwargs.get("use_cache"),
    #             "attention_mask": attention_mask,
    #             "images": kwargs.get("images", None),
    #         }
    #     )
    #     return model_inputs

AutoConfig.register("osprey_mistral", OspreyMistralConfig)
AutoModelForCausalLM.register(OspreyMistralConfig, OspreyMistralForCausalLM)
