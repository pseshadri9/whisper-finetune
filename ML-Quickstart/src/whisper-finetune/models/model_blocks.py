import torch

from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration


class WhisperModelPipeline(torch.nn.Module):
    def __init__(self, model_type: str = "openai/whisper-tiny.en", sr=16000):
        super().__init__()

        self.tokenizer = WhisperTokenizer.from_pretrained(model_type)

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
        self.processor = WhisperProcessor.from_pretrained(model_type)

        self.model = WhisperForConditionalGeneration.from_pretrained(model_type)

    def forward(self, input):
        return self.model(input)
