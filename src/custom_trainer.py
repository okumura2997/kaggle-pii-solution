import datasets
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

from loss import FocalLoss


class WeightedSamplingTrainer(Trainer):
    def set_sampling_weights(self, weights):
        assert len(self.train_dataset) == len(weights), 'sampling weights should have the same length of train dataset'
        self.sampling_weights = weights

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        # weighted sampler
        dataloader_params["sampler"] = WeightedRandomSampler(self.sampling_weights, num_samples=len(train_dataset))

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss_fct = FocalLoss()
        labels = inputs.pop('labels')
        outputs = model(**inputs)

        loss = loss_fct(outputs.logits.view(-1, model.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss