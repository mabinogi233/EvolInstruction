# evolution instructions base on Xun-Fei's Spark AI Model

___

**Now each person can receive 2 million tokens for free on the Xun-Fei Spark AI model(https://xinghuo.xfyun.cn/sparkapi)**



This project use langchain to implement evol-instructions to creating large amounts of instruction data with varying 
levels of complexity using Xun-Fei's Spark AI model.

This tool is refer to **https://github.com/nlpxucan/WizardLM** and **https://arxiv.org/abs/2304.12244**


---

## · how to use it

1. python >= 3.8 (recommend)
2. selenium version > 0.0.266
3. Spark AI Model appid, api_key, api_secret (see https://xinghuo.xfyun.cn/sparkapi)

If you satisfied these requriement, you can copy the source code in a package of your project to use it.

---

## · example


#### first, you need to set the params in instructions_data_evol.py by yourself

```        
# see params in line 24 of instructions_data_evol.py
class Params():
    url = "wss://spark-api.xf-yun.com/v2.1/chat"
    domain = "generalv2"
    app_id = ""  # your appid
    api_secret = "" # your api_secret
    api_key = "" # your api_key
    max_tokens = 1024
    temperature = 0.5

```

#### second, you can use it like this

```
if __name__ == '__main__':

    orign_instruction_save_path = r"" # the path of your orign instruction data
    save_path = r"" # the path of your save instruction data

    e = Evol()
    e.inner_loop(
        max_evol_deep=5, # the max deep of evol
        max_evol_step=30, # the max step of evol
        load_path=orign_instruction_save_path,
        save_path=save_path,
    )
```

#### third, you can implement load and save function by your dataset file



```
see save_instructions(line: 449) and load_instructions(line: 463)

and register your function by 
register_load_function and register_save_function
```

---

## · disclaimer

Everything in this project is affected by the uncertainty of the large language model and is only used for research. This project does not assume any legal responsibility for any adverse effects caused by its use for various purposes.
