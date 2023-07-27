# ADEPT: Adapter-based Efficient Prompt Tuning

**Code for the paper**: ADEPT: Adapter-based Efficient Prompt Tuning approach for smaller language models presented at the 5th Clinical Natural Language Processing Workshop at ACL 2023, Toronto, Canada. 

**Abstract**: Fine-tuning large pre-trained models for downstream tasks can be really expensive. In the past, researchers have proposed various alternatives like adapter and prompt-based methods for tuning these large language models using minimal parameters. However, applying prompt-tuning for smaller language models has not been effective so far and not much work is done in pushing forward soft prompting for these smaller models. To improve the training efficiency of the language models and reduce the size of tuned parameters, we propose a novel **Adapter-based Efficient Prompt Tuning approach (ADEPT)**. In this paper, we show that tuning the parameters of soft prompts with adapter modules while keeping the rest of the model frozen can be a promising method to optimize smaller language models for downstream tasks. Our method achieves up to **98% performance** of full fine-tuning while using only **0.02% of total model parameters**.

If you find our paper useful, please cite us using the bib file:

```
@inproceedings{shah2023adept,
  title={Adept: Adapter-based efficient prompt tuning approach for language models},
  author={Shah, Aditya and Thapa, Surendrabikram and Jain, Aneesh and Huang, Lifu},
  booktitle={Proceedings of The Fourth Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)},
  pages={121--128},
  year={2023}
}
```
