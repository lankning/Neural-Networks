import paddlehub as hub
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
module = hub.Module(name="ernie_gen_couplet")

test_texts = ["十万年寰宇一瞬"]
results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
for result in results:
    print(result)