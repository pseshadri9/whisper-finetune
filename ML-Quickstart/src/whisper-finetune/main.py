import logging

import hydra
import torch
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
#from manifest_sns import SNS
from omegaconf import DictConfig
from transformers import Seq2SeqTrainer
from datasets import Dataset, Audio, DatasetDict

from data_utils.datamodule import AudioDataModule
from data_utils.collate import WhisperCollate
from models.model_blocks import WhisperModelPipeline
from models.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="/home/ubuntu/whisper-finetune/ML-Quickstart/config/whisper-finetune/", config_name="train.yaml")
def main(config: DictConfig):
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")
    #torch.set_num_threads(4)

    log.info(
        f"Pre-processing files from <{config.data.preprocess.source_path}> to \
            <{config.data.preprocess.destination_path}>"
    )
    metadata = hydra.utils.instantiate(config.data.preprocess)
    config.data.datamodule.metadata = metadata

    log.info(f"Instantiating datamodule <{config.data.datamodule._target_}> from {config.data.datamodule.metadata}")
    data = hydra.utils.instantiate(config.data.datamodule)

    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    log.info(f"Instantiating trainer with logger: <{config.trainer.logger._target_}>")
    training_args = hydra.utils.instantiate(config.trainer.training_args)

    collate_fn = WhisperCollate(
    processor=model.processor,
    decoder_start_token_id=model.model.config.decoder_start_token_id,
   )

    #test = hf_dataset(data.test_dataset)
    #raise Exception(test.features)
    model = model.cuda()

    d = hf_dataset(data.train_dataset, data.test_dataset, model.feature_extractor, model.tokenizer)
    trainer = Seq2SeqTrainer(
    args=training_args,
    model=model.model,
    train_dataset= d['train'],
    eval_dataset= d['test'],
    data_collator=collate_fn,
    compute_metrics=compute_metrics(model.tokenizer),
    tokenizer=model.processor.feature_extractor,
    )

    
    #train_and_evaluate(model, trainer, data, ckpt_path=config.ckpt)
    print(trainer.evaluate())
    trainer.train()

def hf_dataset(train_dataset, test_dataset, feature_extractor, tokenizer):
        """Creates a Hugging Face Dataset that lazily indexes the PyTorch dataset."""
        def generator_train():
            for i in range(len(train_dataset)):
                yield train_dataset[i]
        
        def generator_test():
            for i in range(len(test_dataset)):
                yield test_dataset[i]
        
        def process_audio(batch):
            # load and resample audio data from 48 to 16kHz
            audio = batch["audio"]

            # compute log-Mel input features from input audio array
            batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

            # encode target text to label ids 
            batch["labels"] = tokenizer(batch["labels"]).input_ids
            return batch
        
        d = DatasetDict()
        d['train'] = Dataset.from_generator(generator_train)
        d['test'] = Dataset.from_generator(generator_test)
        #d = d.cast_column("audio", Audio(sampling_rate=16000))
        return d.map(process_audio, num_proc=4)


if __name__ == "__main__":
    main()
