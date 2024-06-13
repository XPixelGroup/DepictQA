import torch
from transformers import StoppingCriteria


class DepictQAStop(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        super().__init__()
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.stop_flag = [0] * input_ids.shape[0]
        self.start_len = input_ids.shape[1]

    def check_stop(self, output_ids):
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0] :]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids, scores, **kwargs):
        flag = 1
        for idx, output_id in enumerate(output_ids):
            if self.stop_flag[idx] == 1:
                continue
            if self.check_stop(output_id.unsqueeze(0)):
                self.stop_flag[idx] = 1
            else:
                flag = 0
        if flag == 1:
            return True
        return False
