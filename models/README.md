Pretrained models can be downloaded from huggingface
https://huggingface.co/tornet-ml/tornado_detector_baseline_v1

or accessed using the huggingface api:  (assumes `tornet` is in your path)

```python
from huggingface_hub import hf_hub_download
trained_model = hf_hub_download(repo_id="tornet-ml/tornado_detector_baseline_v1", 
                                filename="tornado_detector_baseline.keras")
model = keras.saving.load_model(trained_model,compile=False)
```
