# LlamaSentimentClassification

Here’s a structured **tabular report** summarizing your **LoRA training and evaluation results**:

### **QLoRA Training Report - LLaMA2-7B**

| **Category**          | **Details** |
|----------------------|------------|
| **Base Model**       | LLaMA2-7B (6.65B params) |
| **Quantization**     | 4-bit (BitsandBytes) |
| **Quant Type**       | `nf4` (Normalized Float 4) |
| **Compute Dtype**    | `float16` |
| **Double Quantization** | `false` |
| **Cache Usage**      | `false` |

### **LoRA Configuration**
| **Parameter**         | **Value**  |
|----------------------|------------|
| **LoRA Rank (r)**    | 16 |
| **LoRA Alpha**       | 64 |
| **LoRA Dropout**     | 0.1 |
| **Bias**             | `none` |
| **Task Type**        | `SEQ_CLS` |
| **Targeted Modules** | `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj` |

### **Training Efficiency**
| **Metric**             | **Value**  |
|----------------------|------------|
| **Trainable Parameters** | **12,288** |
| **Total Parameters**     | **6,647,345,152** |
| **Trainable %**         | **0.0002%** |

### **Evaluation Metrics**
| **Metric**                 | **Value**  |
|---------------------------|------------|
| **Eval Loss**             | 3.3867 |
| **Eval Accuracy**         | **62.76%** |
| **Eval AUC**              | **0.8048** |
