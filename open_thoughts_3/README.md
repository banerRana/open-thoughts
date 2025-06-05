## OpenThoughts3-1.2M

The OpenThoughts3-1.2M dataset was used to train the OpenThinker3-7B model. The dataset can be found on [HuggingFace](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M). It is a combination of 850k math samples, 250k code samples, and 100k science samples. We generate the data by sourcing questions from various datasets, processing them through our pipeline and annotating them using QwQ-32B.

<picture>
    <!-- <source media="(prefers-color-scheme: light)" width="100%" srcset="../images/openthoughts3-diagram.png"> -->
    <img alt="OpenThoughts3 Data Curation Recipe" width="100%" src="../images/openthoughts3-diagram.png">
</picture>

Further details on this dataset can be found in our [paper](https://arxiv.org/abs/2506.04178) and in our [blog post](https://www.open-thoughts.ai/blog/ot3).

We will release the code framework we used to create our dataset at this repository.