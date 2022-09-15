import numpy as np
import re


class PromptParser:
    def __init__(self, model):
        self.model = model
        self.regex = re.compile(r'\[.*?\]|.+?(?=\[)|.*')

    def get_prompt_guidance(self, prompt, steps, batch_size):
        prompts = self.parse_prompt(prompt, steps)
        prompt_guidance = np.empty(steps, dtype=object)
        cg = None

        index = 0
        next_step = 0

        for i in range(0, steps):
            if i == next_step:
                _, text = prompts[index]
                print(f"Swapping at step {i} to: {text}")
                cg = self.model.get_learned_conditioning(batch_size * [text])

                index += 1

                if index < len(prompts):
                    next_step, _ = prompts[index]


            prompt_guidance[i] = cg

        return prompt_guidance

    def __parse_float(self, text):
        try:
            return float(text)
        except ValueError:
            return 0.

    def __parse_swap_statement(self, statement):
        fields = str.split(statement[1:-1], ':')
        if len(fields) < 2:
            return "", "", 0.

        if len(fields) == 2:
            return "", fields[0], self.__parse_float(fields[1])
        else:
            return fields[0], fields[1], self.__parse_float(fields[2])


    def __get_step(self, token, steps):
        _, _, weight = token
        if weight >= 1.:
            return int(weight)
        else:
            return int(weight * steps)

    def parse_prompt(self, prompt, steps):
        tokens = self.__get_tokens(prompt)
        values = np.array([self.__get_step(token, steps) for token in tokens if type(token) is tuple])
        values = np.concatenate(([0], values))
        values = np.sort(np.unique(values))

        builders = [(value, list()) for value in values]

        for token in tokens:
            if type(token) is tuple:
                for value, text in builders:
                    word1, word2, _ = token
                    step = self.__get_step(token, steps)
                    text.append(word1 if value < step else word2)
            else:
                for _, text in builders:
                    text.append(token)

        return [(value, ''.join(text)) for value, text in builders]

    def __get_tokens(self, prompt):
        parts = self.regex.findall(prompt)
        result = list()

        for part in parts:
            if len(part) == 0:
                continue

            if part[0] == '[':
                result.append(self.__parse_swap_statement(part))
            else:
                result.append(part)

        return result


