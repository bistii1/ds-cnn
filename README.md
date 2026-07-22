# DS-CNN Keyword Spotting Model

TensorFlow DS-CNN artifacts for 30-keyword spotting (nRF5340 path).

| File | Description |
| --- | --- |
| `kws_tf_float.tflite` | Float TFLite model (~92% test acc) |
| `kws_tf.json` | Labels, MFCC params, feature norm, test accuracy |

Partner may quantize from `kws_tf_float.tflite` using metadata in `kws_tf.json`.
