GAN-based face generation model for realistic Indian facial features to improve inclusivity in AI

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)

A Generative Adversarial Network (GAN) model trained on diverse Indian facial features to address under-representation in synthetic face generation.



📜 Abstract  
Generative Adversarial Networks (GANs) have remarkably succeeded in synthesizing realistic human faces, but existing datasets like CelebA, dominated by Western facial features, fail to represent diverse ethnicities, including Indian faces. Such under-representation limits the inclusivity and applicability of GAN models, reinforcing systemic biases in AI-generated imagery and diminishing their relevance to diverse contexts.  

This project develops a GAN model trained on a dataset of Indian faces from diverse regions and cultures, leveraging **progressive growing** and **spectral normalization** techniques to generate high-quality, demographically inclusive images. By prioritizing representative datasets, we aim to:  
- Reduce bias in AI-generated facial imagery  
- Enhance applicability to Indian and other under-represented demographics  
- Contribute to the discourse on diversity in generative AI  



🚀 Features  
- **Ethnically Balanced Dataset**: Curated Indian faces covering regional diversity.  
- **Stable Training**: Spectral normalization + progressive growing GANs.  
- **Text-to-Face Synthesis**: BERT embeddings for prompt-based generation.  
- **Bias Metrics**: Quantify diversity in generated samples.  



## 🛠️ Installation  
```bash
git clone https://github.com/your-username/indian-face-gan.git
cd indian-face-gan
pip install -r requirements.txt
