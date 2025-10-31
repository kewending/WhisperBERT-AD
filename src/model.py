import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import copy
from src.otke import OTKernel

def build_classifier_head(input_dim, dropout=0.1, num_labels=2, hidden_factors=[0.5, 0.5]):
    """
    Build a generic MLP classifier head.
    
    Args:
        input_dim (int): Dimension of the input features.
        dropout (float): Dropout probability between layers.
        num_labels (int): Number of output classes (2 for binary).
        hidden_factors (list of float): Scaling factors relative to input_dim for hidden layers.
                                         e.g. 0.5 -> [input_dim, input_dim//2]

    Returns:
        nn.Sequential: Classifier head.
    """
    layers = [nn.LayerNorm(input_dim)]
    prev_dim = input_dim

    # hidden layers
    for factor in hidden_factors:
        hidden_dim = int(input_dim * factor)
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        ])
        prev_dim = hidden_dim

    # final output
    layers.append(nn.Linear(prev_dim, num_labels))
    
    return nn.Sequential(*layers)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        normalized = self.norm(x)
        sublayer_output = sublayer(normalized)
        
        if isinstance(sublayer_output, tuple):
            sublayer_result, extra_output = sublayer_output
            return x + self.dropout(sublayer_result), extra_output
        else:
            return x + self.dropout(sublayer_output)

class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=768, num_heads=12, batch_first=True, dropout=0.1, use_ffn=True):
        super().__init__()
        self.use_ffn = use_ffn  
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        self.d_model = d_model
        
        # Conditionally initialize the feed forward layer and sublayer connection based on the use_ffn flag
        if self.use_ffn:
            self.feed_forward = PositionwiseFeedForward(d_model, d_ff=self.d_model*4)
            self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        else:
            self.sublayer = clones(SublayerConnection(d_model, dropout), 1)

    def forward(self, x, y):
        # Define the cross-attention function that will be called by sublayer
        def apply_cross_attention(normalized_x):
            attn_output, attn_weights = self.cross_attention(normalized_x, y, y)
            return attn_output, attn_weights
        
        # Apply cross-attention through sublayer connection (only computed once)
        x, attn_weights = self.sublayer[0](x, apply_cross_attention)
        
        # Conditionally apply the second sublayer and feed-forward if use_ffn is True
        if self.use_ffn:
            x = self.sublayer[1](x, self.feed_forward)

        return x, attn_weights

class BaseClassifier(nn.Module):
    def __init__(self, input_dim, num_labels=2, dropout=0.1, device="cuda", hidden_factors=[0.5]):
        super().__init__()
        self.num_labels = num_labels
        self.classifier = build_classifier_head(
            input_dim=input_dim,
            dropout=dropout,
            num_labels= num_labels,  # binary uses sigmoid
            hidden_factors=hidden_factors
        )
        self.device = device
    
    def forward(self, *args, **kwargs):
        embeddings = self.encode(*args, **kwargs)   # subclass implements this
        logits = self.classifier(embeddings)
        return logits

    def encode(self, *args, **kwargs):
        """To be implemented by subclass"""
        raise NotImplementedError

class TextClassifier(BaseClassifier):
    def __init__(self, text_encoder="bert-base-uncased", **kwargs):
        config = AutoConfig.from_pretrained(text_encoder)
        self.embed_dim = config.hidden_size
        super().__init__(input_dim=self.embed_dim, **kwargs)
        self.bert = AutoModel.from_pretrained(text_encoder, output_hidden_states=True)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Enable gradient checkpointing for the transformer models
        self.bert.gradient_checkpointing_enable()
    
    def encode(self, text_input_ids, text_attention_mask):
        text_output = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_last_hidden_state = text_output.last_hidden_state
        return text_last_hidden_state.mean(dim=1)
    
class AudioClassifier(BaseClassifier):
    def __init__(self, audio_encoder="openai/whisper-small", **kwargs):
        config = AutoConfig.from_pretrained(audio_encoder)
        self.embed_dim = config.hidden_size
        super().__init__(input_dim=self.embed_dim, **kwargs)
        self.whisper = AutoModel.from_pretrained(audio_encoder).encoder
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Enable gradient checkpointing for the transformer models
        self.whisper.gradient_checkpointing_enable()
    
    def encode(self, audio_input_features):
        batch_size = len(audio_input_features)
        pooled = []
        for i in range(batch_size):
            audio_input_values = torch.tensor(audio_input_features[i]).to(self.device)
            audio_output = self.whisper(input_features=audio_input_values).last_hidden_state
            audio_output = audio_output.view(-1, self.embed_dim).unsqueeze(0)
            pooled.append(audio_output.mean(dim=1))
        return torch.cat(pooled, dim=0)

class AudioOTKEClassifier(BaseClassifier):
    def __init__(self, audio_encoder="openai/whisper-small", **kwargs):
        config = AutoConfig.from_pretrained(audio_encoder)
        self.embed_dim = config.hidden_size
        super().__init__(input_dim=self.embed_dim, **kwargs)
        self.whisper = AutoModel.from_pretrained(audio_encoder).encoder

        self.otk_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            OTKernel(in_dim=self.embed_dim, out_size=512, heads=1),
            nn.LayerNorm(self.embed_dim)
        )
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Enable gradient checkpointing for the transformer models
        self.whisper.gradient_checkpointing_enable()

    def encode(self, audio_input_features):
        batch_size = len(audio_input_features)
        pooled = []
        for i in range(batch_size):
            audio_input_values = torch.tensor(audio_input_features[i]).to(self.device)
            audio_output = self.whisper(input_features=audio_input_values).last_hidden_state
            audio_output = audio_output.view(-1, self.embed_dim).unsqueeze(0)
            audio_output = self.otk_layer(audio_output)
            pooled.append(audio_output.mean(dim=1))
        return torch.cat(pooled, dim=0)

class BaseAudioTextModel(nn.Module):
    def __init__(self, audio_encoder="openai/whisper-small", text_encoder="bert-base-uncased", 
                 dropout=0.1, num_labels=2, device="cuda"):
        super().__init__()
        self.device = device
        self.whisper = AutoModel.from_pretrained(audio_encoder).encoder
        self.bert = AutoModel.from_pretrained(text_encoder, output_hidden_states=True)
        self.embed_dim = self.bert.config.hidden_size
        self.dropout = dropout

        self.num_labels = num_labels
        self.classifier = build_classifier_head(
            input_dim=self.embed_dim*2, # concat of text + audio pooled
            dropout=dropout, 
            num_labels=self.num_labels, 
            hidden_factors=[0.5, 0.5]
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.whisper.gradient_checkpointing_enable()
        self.bert.gradient_checkpointing_enable()

    def unfreeze(self, flag=True):
        for param in list(self.bert.parameters()) + list(self.whisper.parameters()):
            param.requires_grad = flag

    def forward_backbones(self, audio_input_features, text_input_ids, text_attention_mask):
        """Common backbone forward pass: returns audio + text hidden states (before pooling)."""
        batch_size = len(audio_input_features)
        audio_hidden_states, text_hidden_states = [], []

        for i in range(batch_size):
            audio_input_values = torch.tensor(audio_input_features[i]).to(self.device)

            text_output = self.bert(
                input_ids=text_input_ids[i].unsqueeze(0),
                attention_mask=text_attention_mask[i].unsqueeze(0)
            )
            text_hidden_states.append(text_output.last_hidden_state)

            audio_output = self.whisper(input_features=audio_input_values).last_hidden_state
            audio_hidden_states.append(audio_output.view(-1, self.embed_dim).unsqueeze(0))

        return audio_hidden_states, text_hidden_states

class BERTWhisper(BaseAudioTextModel):
    def forward(self, audio_input_features, text_input_ids, text_attention_mask):
        audio_states, text_states = self.forward_backbones(audio_input_features, text_input_ids, text_attention_mask)
        embeddings = []

        for a, t in zip(audio_states, text_states):
            text_pooled = t.mean(dim=1)
            audio_pooled = a.mean(dim=1)
            combined = torch.cat([text_pooled, audio_pooled], dim=-1)
            embeddings.append(combined)

        embeddings = torch.cat(embeddings, dim=0)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits

class BERTWhisperOTKE(BaseAudioTextModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.otk_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            OTKernel(in_dim=self.embed_dim, out_size=512, heads=1),
            nn.LayerNorm(self.embed_dim)
        )
    def forward(self, audio_input_features, text_input_ids, text_attention_mask):
        audio_states, text_states = self.forward_backbones(audio_input_features, text_input_ids, text_attention_mask)
        embeddings = []

        for a, t in zip(audio_states, text_states):
            a = self.otk_layer(a)
            text_pooled = t.mean(dim=1)
            audio_pooled = a.mean(dim=1)
            combined = torch.cat([text_pooled, audio_pooled], dim=-1)
            embeddings.append(combined)

        embeddings = torch.cat(embeddings, dim=0)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits

class CrossAttention(BaseAudioTextModel):
    def __init__(self, cross_heads=1, use_ffn=False, **kwargs):
        super().__init__(**kwargs)

        self.cross_attention_audio = CrossAttentionLayer(dropout=self.dropout, num_heads=cross_heads, use_ffn=use_ffn)
        self.cross_attention_text = CrossAttentionLayer(dropout=self.dropout, num_heads=cross_heads, use_ffn=use_ffn)

    def forward(self, audio_input_features, text_input_ids, text_attention_mask):
        audio_states, text_states = self.forward_backbones(audio_input_features, text_input_ids, text_attention_mask)
        embeddings = []

        for a, t in zip(audio_states, text_states):
            text_attended, _ = self.cross_attention_text(t, a)
            audio_attended, _ = self.cross_attention_audio(a, t)

            text_pooled = text_attended.mean(dim=1)
            audio_pooled = audio_attended.mean(dim=1)

            combined = torch.cat([text_pooled, audio_pooled], dim=-1)
            embeddings.append(combined)

        embeddings = torch.cat(embeddings, dim=0)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits

class CrossAttentionOTKE(BaseAudioTextModel):
    """
    Multimodal classifier combining audio (Whisper) + OTKE and text (BERT) encoders with cross-attention mechanism.
    """
    def __init__(self, cross_heads=1, use_ffn=False, **kwargs):
        super().__init__(**kwargs)

        self.cross_attention_audio = CrossAttentionLayer(dropout=self.dropout, num_heads=cross_heads, use_ffn=use_ffn)
        self.cross_attention_text = CrossAttentionLayer(dropout=self.dropout, num_heads=cross_heads, use_ffn=use_ffn)

        self.otk_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            OTKernel(in_dim=self.embed_dim, out_size=512, heads=1),
            nn.LayerNorm(self.embed_dim)
        )

    def forward(self, audio_input_features, text_input_ids, text_attention_mask):

        audio_states, text_states = self.forward_backbones(audio_input_features, text_input_ids, text_attention_mask)
        embeddings = []

        for a, t in zip(audio_states, text_states):
            a = self.otk_layer(a)
            text_attended, _ = self.cross_attention_text(t, a)
            audio_attended, _ = self.cross_attention_audio(a, t)

            text_pooled = text_attended.mean(dim=1)
            audio_pooled = audio_attended.mean(dim=1)

            combined = torch.cat([text_pooled, audio_pooled], dim=-1)
            embeddings.append(combined)

        embeddings = torch.cat(embeddings, dim=0)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits