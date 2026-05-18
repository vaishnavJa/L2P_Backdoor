# Backdoor Poisoning Experiments on L2P

This repository is a research fork of the PyTorch implementation of
**Learning to Prompt for Continual Learning (L2P)**. I originally cloned the
L2P PyTorch codebase and made my thesis changes directly inside it.

The thesis focus of this fork is **backdoor poisoning attacks against
prompt-based continual learning**, with L2P on Split-CIFAR100 as the main
model and benchmark.

Original L2P paper:

> Wang, Zifeng, et al. "Learning to Prompt for Continual Learning." CVPR 2022.

Original PyTorch implementation:

<https://github.com/JH-LEE-KR/l2p-pytorch>

Official JAX implementation:

<https://github.com/google-research/l2p>

## Research Focus

The base L2P method learns a pool of prompts for a frozen pretrained Vision
Transformer and selects prompts dynamically for each input. This fork studies
what happens when poisoned samples and learned triggers are introduced into
that prompt-learning process.

The main questions explored here are:

- Can a trigger cause the L2P model to predict a target class while preserving
  clean continual-learning accuracy?
- How does poisoning affect prompt selection and task-wise performance?
- How does attack success rate (ASR) evolve across continual tasks?
- Can selected poisoned samples or retraining strategies improve attack
  effectiveness?

## Repository Map

| Path | Purpose |
| --- | --- |
| `main.py` | CLI entrypoint. Builds datasets, original frozen ViT, prompt ViT, optimizer, then calls training or evaluation. |
| `engine.py` | Main training, trigger learning, poisoned training, evaluation, checkpointing, ACC matrix, and ASR matrix logic. |
| `backdoor.py` | Trigger application, poisoned sample selection, and poisoned batch-index helpers. |
| `prompt.py` | L2P prompt pool, prompt key similarity, top-k prompt selection, and pull-constraint values. |
| `vision_transformer.py` | ViT implementation modified to insert prompt tokens and classify from prompt-token outputs. |
| `models.py` | Registers ViT model variants with `timm`. |
| `datasets.py` | Continual dataloader construction for Split-CIFAR100 and 5-datasets. |
| `configs/cifar100_l2p.py` | Main config for Split-CIFAR100 L2P and poisoning experiments. |
| `configs/five_datasets_l2p.py` | Config for the original 5-datasets L2P setup. |
| `train_cifar100_l2p.sh` | Slurm/scripted experiment commands used for poisoning runs. |
| `runlogs/` | Saved training/evaluation logs from cluster runs. |
| `bestselect/stats/` | Saved selected poison indices plus ASR/ACC matrices. |
| `trigger_*.pt`, `trigger/*.pt` | Saved learned trigger tensors. |

## Main Training Flow

The normal flow starts in `main.py`:

1. Select a config, usually `cifar100_l2p`.
2. Build continual dataloaders with `build_continual_dataloader`.
3. Create an original pretrained ViT used for frozen feature extraction.
4. Create the L2P ViT with prompt-pool arguments.
5. Freeze most backbone parameters and train prompt/head parameters.
6. Call `train_and_evaluate` in `engine.py`.

For Split-CIFAR100, the default setup uses 10 tasks with 10 classes per task.
After each task, the model is evaluated on tasks seen so far and a checkpoint
is saved.

## L2P Prompt Logic

The L2P implementation is centered around `prompt.py` and
`vision_transformer.py`.

`Prompt.forward` computes an embedding key for each input, compares it against
learned prompt keys, selects top-k prompts, and prepends the selected prompt
tokens to the ViT patch embeddings.

Important prompt options:

- `--prompt_pool`: enables the prompt pool.
- `--size`: number of prompts in the pool.
- `--length`: number of tokens per prompt.
- `--top_k`: number of prompts selected per input.
- `--embedding_key`: feature used for prompt lookup, usually `cls`.
- `--batchwise_prompt`: selects prompts based on the most common batch-level
  prompt choices.
- `--pull_constraint`: encourages selected prompt keys to align with input
  features.

The classifier usually uses `--head_type prompt`, meaning the final prediction
is made from the prompt-token representations rather than only the class token.

## Backdoor Poisoning Logic

Backdoor-specific logic is mainly in `engine.py` and `backdoor.py`.

The attack target is controlled by:

- `--p_task_id`: target task id.
- `--p_class_id`: target class id within that task.
- `--poison_rate`: fraction of samples to poison.
- `--trigger_path`: path used to save or load a trigger tensor.
- `--use_trigger`: whether to load and use an existing trigger.
- `--best_select`: whether to use saved selected poisoning indices.
- `--retrain`: whether to run the retraining attack variant.
- `--black_box`: whether to run the black-box style variant.

For Split-CIFAR100, the target label is currently computed as:

```python
target_label = p_task_id * 10 + p_class_id
```

The trigger is added to images in `poison_dataset` / `poison_target_dataset`.
The learned trigger tensors are full-image tensors with shape
`(1, 3, 224, 224)` in the current code, then clamped before being added to the
input.

## Attack Success Rate

Attack success rate (ASR) is measured on triggered inputs:

```text
ASR = fraction of poisoned samples predicted as the target label
```

Clean accuracy and ASR are tracked separately:

- `Acc@1`: clean task accuracy.
- `Acc@1_b`: clean accuracy on the target class for the target task.
- `ASR`: attack success rate on poisoned inputs.
- `ACC`: accuracy on the non-poisoned part of a poisoned batch.
- `acc_matrix`: task-by-task clean accuracy matrix.
- `asr_matrix`: task-by-task attack success matrix.

Saved matrices appear in paths such as:

```text
bestselect/stats/*acc_matrix*.npy
bestselect/stats/*asr_matrix*.npy
```

## Example Commands

Clean L2P training on Split-CIFAR100:

```bash
python main.py cifar100_l2p \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path ./../local_datasets/ \
  --output_dir ./output
```

Learn or generate a trigger for target task 0, class 0:

```bash
python main.py cifar100_l2p \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path ./../local_datasets/ \
  --output_dir ./output \
  --use_trigger False \
  --poison_rate 0.1 \
  --trigger_path trigger_0_0_0.1_16_255.pt \
  --p_task_id 0 \
  --p_class_id 0
```

Train/evaluate using a saved trigger:

```bash
python main.py cifar100_l2p \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path ./../local_datasets/ \
  --output_dir ./output \
  --use_trigger True \
  --poison_rate 0.1 \
  --trigger_path trigger_0_0_0.1_16_255.pt \
  --p_task_id 0 \
  --p_class_id 0
```

Evaluate saved checkpoints:

```bash
python main.py cifar100_l2p \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path ./../local_datasets/ \
  --output_dir ./output \
  --eval
```

## Environment

The original implementation was tested with:

- Ubuntu 20.04
- Python 3.8
- PyTorch 1.12.1
- torchvision 0.13.1
- timm 0.6.7
- Pillow 9.2.0
- matplotlib 3.5.3

Install the listed Python packages with:

```bash
pip install -r requirements.txt
```

Depending on the machine, PyTorch and torchvision may need to be installed
separately with the correct CUDA build.

## Data

Pass the dataset root with `--data-path`.

The main thesis experiments use Split-CIFAR100. The original L2P 5-datasets
setup is still present, but the poisoning code is most directly tied to
Split-CIFAR100 because of the current target-label calculation.

## Notes for Future Cleanup

The poisoning logic was added directly into the original L2P codebase during
thesis experimentation. The highest-value cleanup work would be:

- Separate clean L2P training, trigger learning, poisoned training, and
  evaluation into clearer commands or modules.
- Generalize target-label handling beyond `p_task_id * 10 + p_class_id`.
- Make trigger size, position, clamp value, and evaluation boost explicit CLI
  arguments.
- Save all stats consistently under `output_dir/stats/`.
- Move large experiment artifacts and triggers out of normal git tracking or
  into Git LFS/DVC.
- Add a compact experiment table describing which trigger files correspond to
  which thesis runs.

## Citation

If you use the original L2P method, cite:

```bibtex
@inproceedings{wang2022learning,
  title={Learning to prompt for continual learning},
  author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={139--149},
  year={2022}
}
```

## License

This repository keeps the Apache 2.0 license from the original L2P PyTorch
implementation. See `LICENSE`.
