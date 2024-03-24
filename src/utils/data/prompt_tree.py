import json

from utils.pkg.dict2xml import dict2xml


class PromptTree:
    def __init__(self, cfg, data, id, name_alias, style='xml', label=None, use_variant_prompt=False):
        prompt = ''
        info_dict = {}        
        self.style = style
        self.label = label
        
        if cfg.use_static_text:  # Add static text as prompt prefix
            prompt += data[id].Static_description + '\n\n'
        else:  # Add static text to info_dict to be further formatted
            info_dict['Static'] = data[id][cfg.data.static_cols].T.squeeze().to_dict()

        if cfg.use_dynamic_text:  # Add static text as prompt prefix
            prompt += data[id].dynamic_prompt
            prompt += '\n\n'
            
        if cfg.use_trends:
            prompt += data[id].Trend_prompt
            prompt += '\n\n'            
        
        if cfg.use_cont_fields:
            prompt += 'The sequential information of hospitalization is:\n'
            for cont_field in cfg.data.dynamic_cols:
                if cfg.use_seq_encoder and cont_field in cfg.in_cont_fields:
                    info_dict[cont_field] = f'<{cont_field.upper()}-EMB>'
                else:
                    info_dict[cont_field] = str(data[id][cont_field])
            if self.style == 'json':
                prompt += json.dumps(info_dict, indent=4)
            elif self.style == 'xml':
                prompt += dict2xml(info_dict, wrap="information", indent="\t")
        else:
            prompt += data[id].hospitalization_per_100k_gpt_trend + '\n\n'
            prompt += data[id].reported_cases_per_100k_gpt_trend
        
        if use_variant_prompt:
            variant_prompt = data[id].variant_prompt
            if variant_prompt:
                prompt += '\n\n'
                prompt += variant_prompt
                prompt += '\n\n'
            
        self.prompt = prompt

    def __str__(self):
        return self.prompt

    def __repr__(self):
        return self.prompt
