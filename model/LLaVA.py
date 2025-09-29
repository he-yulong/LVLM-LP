import sys
import os
import json

import torch
from torch import nn
import numpy as np
from io import BytesIO
from transformers import TextStreamer
from transformers.generation import BeamSearchDecoderOnlyOutput

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from model.base import LargeMultimodalModel

from transformers import LogitsProcessor
class SafetyLogitsProcessor(LogitsProcessor):
    def __init__(self, safety_model_path, tokenizer, refusal_phrase="I cannot help you with that.", threshold=0.5):
        self.threshold = threshold
        self.lr_model = torch.load(safety_model_path, map_location="cpu")
        self.weights = self.lr_model['weights']    # shape: [1, vocab_size] if saved as in your code
        self.bias = self.lr_model['bias']          # shape: [1]
        # Build safe phrase ids WITHOUT adding specials
        self.safe_ids = tokenizer.encode(refusal_phrase, add_special_tokens=False)
        # append eos so decoding will stop cleanly after the phrase
        if tokenizer.eos_token_id is not None:
            self.safe_ids = self.safe_ids + [tokenizer.eos_token_id]
        # runtime state
        self.active = False     # becomes True when probe says "unsafe"
        self.step = 0           # which token of safe_ids we’re forcing next

    def __call__(self, input_ids, scores):
        # If we are already in "active" mode, keep forcing the phrase
        if self.active:
            forced_id = self.safe_ids[min(self.step, len(self.safe_ids)-1)]
            safe_logits = torch.full_like(scores, -float("inf"))
            safe_logits[0, forced_id] = 0.0
            self.step += 1
            return safe_logits

        # Otherwise, run the LR probe on the **current** next-token logits
        next_token_logits = scores  # [1, vocab_size]
        device = next_token_logits.device

        weights = self.weights.to(device)  # [1, vocab_size]
        bias = self.bias.to(device)        # [1]

        # logistic regression score & prob
        lr_logits = torch.matmul(next_token_logits, weights.T) + bias  # [1,1]
        prob = torch.sigmoid(lr_logits).item()

        # If unsafe → activate & force the first token of refusal
        if prob < self.threshold:
            self.active = True
            self.step = 0
            forced_id = self.safe_ids[self.step]
            safe_logits = torch.full_like(scores, -float("inf"))
            safe_logits[0, forced_id] = 0.0
            self.step += 1
            return safe_logits

        # safe: let the model proceed normally
        return scores

USE_DECODING_STRATEGY = False
class LLaVA(LargeMultimodalModel):
    def __init__(self, args):
        super(LLaVA, self).__init__()
        load_8bit = False
        load_4bit = False
        
        # Load Model
        disable_torch_init()

        model_name = get_model_name_from_path(args.model_path)
        if "finetune-lora" in args.model_path:
            model_base = "liuhaotian/llava-v1.5-7b"
        elif "lora" in args.model_path:
            model_base = "lmsys/vicuna-7b-v1.5"
        else:
            model_base = None
        print(model_base)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, model_base, model_name, load_8bit, load_4bit)
        print(model_name)
        self.conv_mode = "llava_v1"
        
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
    
    def refresh_chat(self):
        self.conv = conv_templates[self.conv_mode].copy()
        self.roles = self.conv.roles
    
    def _basic_forward(self, image, prompt, return_dict=False):
        self.refresh_chat()
        
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.unsqueeze(0).half().to(self.device)

        # message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)

        conv_prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer, 
                                          IMAGE_TOKEN_INDEX, 
                                          return_tensors='pt').unsqueeze(0).cuda()
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = ["###"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            if USE_DECODING_STRATEGY:
                from transformers import LogitsProcessorList

                # inside _basic_forward(...)
                logits_processors = LogitsProcessorList()
                safety_model_path = "./output/LLaVA-7B/lr_model_safety_oe.pt"
                if os.path.exists(safety_model_path):
                    print("Safety model loaded: applying safety-aware decoding.")
                    logits_processors.append(
                        SafetyLogitsProcessor(
                            safety_model_path=safety_model_path,
                            tokenizer=self.tokenizer,
                            refusal_phrase="Sorry, I cannot help with that due to safety concerns.",
                            threshold=0.5
                        )
                    )

                outputs = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    logits_processor=logits_processors,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    max_new_tokens=100,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    return_dict_in_generate=return_dict,
                    output_attentions=return_dict,
                    output_hidden_states=return_dict,
                    output_scores=return_dict,
                )
            else:
                outputs = self.model.generate(
                    input_ids,
                    images=image_tensor,

                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,

                    max_new_tokens=100,
                    streamer=streamer,
                    use_cache=True,

                    stopping_criteria=[stopping_criteria],

                    return_dict_in_generate=return_dict,
                    output_attentions=return_dict,
                    output_hidden_states=return_dict,
                    output_scores=return_dict)
            
        return input_ids, outputs
    
    def forward_with_probs(self, image, prompt):
        input_ids, outputs = self._basic_forward(image, prompt, return_dict=True)
        
        if isinstance(outputs, BeamSearchDecoderOnlyOutput):
            beam_indices = outputs['beam_indices'][0].cpu()
            beam_indices = [i for i in beam_indices if i != -1]
            logits = None
            probs = float(outputs['sequences_scores'].cpu().item())
            output_ids = outputs["sequences"][0][-len(beam_indices):]
        else:
            logits = torch.cat(outputs['scores'], dim=0).cpu().numpy()
            probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
            probs = torch.cat(probs).cpu().numpy()
            output_ids = outputs["sequences"][0][len(input_ids):]
            
        # If in conf mode, try to parse numeric value
        if "considering the provided image" in prompt.lower():  # crude check for conf prompt
            print(output_ids)
            response_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            print(f"Raw response: {response_text}")
            try:
                # Extract the first float-like number from the response
                import re
                match = re.search(r"[-+]?\d*\.?\d+", response_text)
                if match:
                    response = float(match.group(0))
                else:
                    print(f"No float-like number found in the response")
                    response = 0.0
                print(f"Parsed confidence: {response}")
            except Exception as e:
                print(f"Parse error: {e}")
                response = 0.0
        else:
            response = self.tokenizer.decode(output_ids).strip()[:-4]

        output_ids = output_ids.cpu().numpy()
#         output_probs = [probs[i][output_ids[i]] for i in range(len(probs))]
        
        return response, output_ids, logits, probs