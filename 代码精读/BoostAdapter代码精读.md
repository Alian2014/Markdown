![BoostAdapter算法流程图](BoostAdapter算法流程图.png)

---

## 一、图像增强模块（左上）
- **模块**：get_ood_preprocess(args)，get_cross_dataset_preprocess(preprocess, args)

- **作用**：OOD数据集/跨数据集的图像增强器，返回一个可以call的类对原图像进行增强，返回原图+增强图的列表，增强配置来源于args

- **两个增强模块的区别**：base_transform有区别，get_ood_preprocess(args)进行放缩裁剪，get_cross_dataset_preprocess(preprocess, args)不进行放缩裁剪

---

## 二、获取类文本向量矩阵（左下）

- **模块**：clip_classifier(...)

- **作用**：根据类名称和人工提示构造类文本向量矩阵，用于后续与图像特征矩阵做积

---

## 三、获取图像特征向量（左中）

- **模块**：get_clip_logits(...)

- **作用**：对输入图像计算 CLIP 模型的特征向量、logits，以及相关的预测信息（如 loss、prob_map、pred）

---

## 四、Filter by Entropy（右下）

- **模块1**：get_clip_logits(...)

- **作用**：get_clip_logits(...) 选出最有信心（熵最小）的前 10% 样本，用它们做特征聚合，构建 Boosting Distribution。同时返回原图特征和 logits

- **模块2**：get_entropy(...)

- **作用**：把实际的熵 loss 除以理论最大熵，得到一个比例值 ∈ [0, 1]。比例越高，说明模型预测越不确定;比例越低，说明模型预测越 confident。

---

## 五、