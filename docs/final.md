---
layout: default
title: Final Report
---
# {{ page.title }}
Final Report Video click [here](https://youtu.be/w5B54gXPq4U)

<iframe width="560" height="315"
src="https://www.youtube.com/embed/w5B54gXPq4U
title="Status report video"
frameborder="0"
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
allowfullscreen>
</iframe>  

# Project Summary

Our project uses Reinforcement Learning to fine-tune a large language model in the text-to-sql task. It investigates whether reinforcement learning can improve a language model’s performance on text-to-SQL without requiring a much larger model. This could support the development of resource-constrained LLMs that are cheaper to run, easier to deploy, and still highly effective on specialized tasks such as translation or database querying.

We chose to focus on converting English descriptions to SQL because it highlights a major challenge in machine learning: bridging the gap between human natural language and rigid SQL logic. Natural human language is inherently ambiguous; a user might ask the same question in dozens of different ways, using vague terms or implied context. In contrast, SQL requires strict, exact syntax where a single incorrect column name or missing keyword causes the entire query to fail. This clash between flexible English and strict SQL makes the task much harder than general text generation, even for larger models with the table schema provided. Reinforcement learning for SQL has a long history stretching back before LLMs to schema-aware encoders and grammar-based models, making it an appropriate testbed for our approach.

To explore what affects final model performance, we evaluated various reward designs as well as system prompts. In addition, we explore how our models with their limited compute compare to other models that are more expensive to run. 
In addition to creating the pipeline, we experimented with two weight scheduling programs throughout the quarter. Near the end, we also completed additional model comparison tests on query complexity by splitting queries by token length. 

# Approach
We decided to use the GRPO algorithm to refine our LLMs with verifiable rewards, as it has gained popularity within recent times for its performance. It also affords us the benefit of not needing to learn a separate critic model during training, which is difficult to do and requires additional computational resources.

$$\hat{A}_{i,t} = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}$$

$$
\begin{aligned}
\mathcal{L}_{\mathrm{GRPO}}(\theta)
&= -\frac{1}{\sum_{i=1}^{G} |o_i|}
\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\Bigg[
\min\Bigg(
\frac{\pi_{\theta}(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q, o_{i,<t})}\hat{A}_{i,t}, \\
&\qquad
\mathrm{clip}\Bigg(
\frac{\pi_{\theta}(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q, o_{i,<t})},
1-\epsilon, 1+\epsilon
\Bigg)\hat{A}_{i,t}
\Bigg)
{} - \beta \mathbb{D}_{\mathrm{KL}}[\pi_{\theta}\|\pi_{\mathrm{ref}}]
\Bigg]
\end{aligned}
$$

The loss function is set up such that the model learns to produce complete answers that rank higher in comparison to other completions it has made. In order to improve stability, it makes sure that the resulting change is not too sharp by clipping the reward signal. In addition, the loss includes a KL divergence term so that the fine-tuned model does not stray too far from the pretrained model.

### Data Usage
We trained the model using the Spider 1.0 dataset, utilizing approximately 9,000 data points for the training phase. We set the training batch size to 6 and the maximum completion length to 512 tokens. A fixed random seed of 42 was used to ensure the setup is reproducible. Additionally, evaluations were scheduled to run on a random subset of 512 data points every 512 steps on a fixed seed.

For this specific application, the state is the natural language question paired with the relevant database schema provided in the prompt. The action is the generated SQL query, enclosed within the ```<sql>...</sql>``` tags. The reward functions are multifaceted and dynamic, employing a ScheduledReward class that adjusts the weights of different reward functions based on the training progress. These components evaluate the generated SQL on several criteria.


### Reward Structure
__Syntax Check Reward__: Execute the generated query against the corresponding table, returning 1.0 if the SQL executes without errors and 0.0 otherwise. This ensures basic SQL syntax requirements are met and the model generates the query within the <sql>...</sql> tags.

__Schema Linking Reward__: Extracts schema items from both the gold query and the predicted query and calculates the Jaccard similarity between the tables and columns used in both sets. Return a score in the range of 0.0-1.0 based on that.

__Query N-gram Comparison Reward__: Tokenize the gold query and the predicted query, then compute the SequenceMatcher ratio between the two lists and return a score in the range of 0.0-1.0 based on that.

__Comprehensive Execution Reward__: Based on the results from executing both the gold query and the predicted query, calculate a combined score heavily weighted towards execution accuracy. The score comprises a Column Intersection Score (measuring structural accuracy of the output table) and a Row Content F1 Score (measuring data accuracy against the gold execution results). Based on those two scores, return a floating point in the range of 0.0 -1.0.

We experimented with scheduling a weight on these rewards throughout training to see if we could improve the model’s performance. This was done by giving more importance to easier tasks at the beginning of training and shifting this importance to the more difficult ones as training finishes. Specifically, we implemented a piecewise linear scheduler to adjust the reward weights as training progresses, as shown below.

<img src="{{ '/weight_scheduling.png' | relative_url }}" alt="Weight Scheduling 1.0">

<img src="{{ '/weight_scheduling2.png' | relative_url }}" alt="Weight Scheduling 2.0">
 

# Evaluation


__Execution Exact Match Reward__: Execute the gold query and the predicted query against the corresponding table. Return a 1.0 if both sets of results match exactly and 0.0 otherwise.

__Execution Subset Match Reward__: Execute the gold query and the predicted query against the corresponding table. Return the jaccard similarity of the two sets of results in the range of 0.0 - 1.0.

__Comprehensive Execution Reward__: Based on the results from executing both the gold query and the predicted query, calculate a combined score heavily weighted towards execution accuracy. The score comprises a Column Intersection Score (measuring structural accuracy of the output table) and a Row Content F1 Score (measuring data accuracy against the gold execution results). Based on those two scores, return a floating point in the range of 0.0 -1.0.

Although the execution subset and comprehensive execution rewards are helpful when understanding how the model is performing during training, we decided to use the execution exact match reward when evaluating our trained models. This decision was made as this reward function measures the raw performance that would be important for real world tasks.

### Model Comparison
--------------------

We first compare three different models. Qwen2.5-0.5coder and Qwen3-0.6b are similarly sized models with the former trained specifically for coding tasks while the latter is newer. We additionally include deepseek-coder-1.3b which is more than twice the size of the Qwen models and was also pretrained for coding tasks.

<img src="{{ '/model_comparison.png' | relative_url }}" alt="Model Comparison">
***deepseek: Blue, Qwen3: Orange, Qwen2.5: Pink***

It can be seen that deepseek achieves 6-12% better performance when compared to our Qwen models. We noticed that although Qwen3 was not pretrained for coding, it ended up out-performing Qwen2.5-coder which went against our intuition that pre-training in the general sphere of a particular task can help improve model performance for said task.  We investigated this by fine-tuning the vanilla Qwen2.5 model using our methods.

<img src="{{ '/qwen2.5_comparison.png' | relative_url }}" alt="Qwen2.5 Comparison">
***Coding Pre-training: Pink, No Pre-Training: Green

Here we found that the coding pretraining does in fact help with model performance. We attribute the difference found between the performance within Qwen2.5-Coder and Qwen3 to the updates that were made between the two versions. We would have explored if a pretrained Qwen3-coder would have had additional performance, but there was no model available in a size that we needed.

### Weight Scheduling
--------------------

Additionally we investigated if weight scheduling improves model performance.

<img src="{{ '/deepseek_schedule.png' | relative_url }}" alt="Deepseek Reward Schedule Comparison">
***All Reward Schedules for deepseek***

<img src="{{ '/qwen2.5_schedule.png' | relative_url }}" alt="Qwen2.5 Reward Schedule Comparison">
***All Reward Schedules for Qwen2.5***

<img src="{{ '/qwen3_schedule.png' | relative_url }}" alt="Qwen3 Reward Schedule Comparison">
***All Reward Schedules for Qwen3***

We created two weight schedule programs over the quarter. It was found that our weight schedules had no significant impact. This could be a part of the training process that would need tuning in order to see results, but additional experiments would need to be required.

### Reward Functions
--------------------

We addtionally investigated how our reward functions affect model performance. Although we did not have time to do a full ablation study, we were able to investigate two specific rewards.

We first chose to see if removing the schema linking reward would impact performance. We noticed that during training from start to finish that this reward would have a fairly consistent value of one. This indicated that the model could have been strong within this specific task and the reward was redundant. 

<img src="{{ '/qwen2.5_schema_linking.png' | relative_url }}" alt="Qwen2.5 No Schema Comparison">
***Qwen2.5: With Schema Linking (bigger), Pink. Without Schema Linking, Grey***

<img src="{{ '/qwen3_schema_linking.png' | relative_url }}" alt="Qwen3 No Schema Comparison">
***Qwen3: With Schema Linking (bigger), Orange. Without Schema Linking, Light Pink***

<img src="{{ '/deepseek_schema_linking.png' | relative_url }}" alt="Deepseek No Schema Comparison">
***Deepseek: With Schema Linking (bigger), Dark Blue. Without Schema Linking, Light Orange***

However, for every model it was seen that our intuition in regards to how this reward affects our models performance was incorrect.

The other reward we experimented with was the execution exact match reward. Although we evaluate our models using this reward, we never train with it. Therefore, we wanted to see if including it could help with our final model performance

<img src="{{ '/qwen2.5_exact.png' | relative_url }}" alt="Qwen2.5 With Exact Match">
***Qwen2.5: With Exact Match (bigger), Light Orange. Without Exact Match, Pink***

<img src="{{ '/qwen3_exact.png' | relative_url }}" alt="Qwen3 With Exact Match">
***Qwen3: With Exact Match (bigger), Blue. Without Exact Match, Orange***

<img src="{{ '/deepseek_exact.png' | relative_url }}" alt="Deepseek With Exact Match">
***Deepseek: With Exact Match (bigger), Green. Without Exact Match, Dark Blue***

We noticed that including this reward had small but negligible gains in performance.

### Qualitative Results

In order to better understand our qualitative results we investigated how query complexity affects our model performance. We measured this by splitting our test data by how many tokens the correct sql-query has.

<img src="{{ '/complex_deepseek.png' | relative_url }}" alt="Complexity Bar Chart for deepseek">

<img src="{{ '/complex_qwen2.5.png' | relative_url }}" alt="Complexity Bar Chart for Qwen2.5">

<img src="{{ '/complex_qwen3.png' | relative_url }}" alt="Complexity Bar Chart for Qwen3">

We saw across all models that an increase in complexity caused a decrease in performance. For the most complex 20% of queries, Deepseek achieved more than double the accuracy of Qwen 2.5 and Qwen 3.0.

### Example Output from deepseek
<img src="{{ '/correct_output.png' | relative_url }}" alt="Correct Model Output">

<img src="{{ '/correct_actual.png' | relative_url }}" alt="Correct Gold">
***The model output query (top) matches the gold query(bottom)***

<img src="{{ '/wrong_output.png' | relative_url }}" alt="Wrong Model Output">

<img src="{{ '/wrong_actual.png' | relative_url }}" alt="Wrong Output Actual Answer">
***The model hallicinates the column "customer_id" (top) which does not match the gold query(bottom)*** 


This difference in performance in relation to the query complexities was noticeable when interacting with the model directly. In this specific example our trained deepseek model experienced no difficulty in generating the correct SQL code for the simple query but contained small but numerous mistakes for the more complex one.

# Resource Used

ChatGPT has been used within the project in order to check understanding of GRPO, correct grammatical errors within the writeup, and correct slurm usage.

DeepLearn Library was used for acceleration and optimization during training. Specifically, the DeepLearn stage 1 tune got applied to split the model among GPUs to avoid out-of-memory problems.

HuggingFace was used to create a trainer using ```GRPOConfig``` and ```GRPOTrainer```
