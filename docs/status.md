---
layout: default
title:  Status
---

# {{ page.title }}


## Evaluations


## Remaining Goals and Challenges
Our goal at a high level is to improve our current text-to-SQL pipeline into a better evaluated system that shows clear improvement over a baseline model on Spider. We plan to continue iterating on reward design and test different preprocessing strategies that improve results. We will continue comparing how different base models respond to the same GRPO setup. We also want to expand our evaluation with multiple concurrent evaluations execution accuracy and exact match, to get more details about helpful changes.  

One of the challenges is that when reward tuning, a query can be logically correct but look different from the reference, being marked wrong. This is an issue specific to query scoring that we hope to fix with more advanced evaluation scoring reward functions. We also face practical constraints around compute, training time, and debugging to slow iteration, especially as we test multiple reward functions and model variants.  

## Resources Used