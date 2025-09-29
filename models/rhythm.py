import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # Basic initialization remains the same
        self.token_len = configs.token_len
        
        # Device configuration
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
            
        # Load LLaMA
        self.llama = AutoModelForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            token=configs.token
        )
        
        # Freeze LLaMA
        for param in self.llama.parameters():
            param.requires_grad = False
            
        self.hidden_dim = self.llama.config.hidden_size
        
        # Embedding dimensions
        user_embeds_size = configs.user_embeds_size
        times_embeds_size = configs.times_embeds_size
        latlon_emb_dim = configs.latlon_emb_dim
        place_embeds_size = configs.place_embeds_size
        drop_rate = configs.drop_rate
        
        self.token_len = configs.token_len
        self.token_num = configs.token_num
        
        # Temporal embeddings
        self.user_embed = nn.Embedding(configs.num_users, user_embeds_size)
        self.tod_embed = nn.Embedding(48, times_embeds_size)
        self.dow_embed = nn.Embedding(7, times_embeds_size)
        
        # Spatial embeddings
        self.place_embed = nn.Embedding(configs.num_classes, place_embeds_size)
        self.latlon_proj = nn.Sequential(
            nn.Linear(6, latlon_emb_dim),
            nn.GELU(),
            nn.LayerNorm(latlon_emb_dim)
        )
        
        # Calculate embedding dimensions
        self.temporal_emb_dim = user_embeds_size + times_embeds_size * 2  # 384
        self.spatial_emb_dim = latlon_emb_dim + place_embeds_size  # 384
        
        # Projection to LLaMA hidden dimension
        self.temporal2hidden = nn.Sequential(
            nn.Linear(self.temporal_emb_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        
        self.spatial2hidden = nn.Sequential(
            nn.Linear(self.spatial_emb_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        
        self.full_embed_mlp = ResidualMLPBlock(self.hidden_dim, drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.pool_proj = PoolingModule(
            hidden_dim=self.hidden_dim,
            seq_len=self.token_len,
            pool_type='linear',
            dropout=drop_rate
        )
        self.num_places = configs.num_classes
        self.classifier = ProjectionLayer(self.hidden_dim, self.num_places)
        self.hierarchical_attention = HierarchicalAttention(self.hidden_dim)
        self.segment_attention = MultiLayerAttentionEncoder(self.hidden_dim, num_layers=configs.num_layers)
        
        # Learnable parameters for prompt fusion
        self.prompt_fusion1 = nn.Parameter(torch.tensor(0.0))
        self.prompt_fusion2 = nn.Parameter(torch.tensor(0.0))
        
    def encode_temporal(self, user_id, tod, dow):
        """
        Encode temporal features (user, time of day, day of week)
        
        Args:
            user_id (torch.Tensor): User IDs [B, T]
            tod (torch.Tensor): Time of day [B, T]
            dow (torch.Tensor): Day of week [B, T]
            
        Returns:
            torch.Tensor: Temporal embeddings [B, T, hidden_dim]
        """
        # Get embeddings
        emb_user = self.user_embed(user_id)  # [B, T, user_embeds_size]
        emb_tod = self.tod_embed(tod)        # [B, T, times_embeds_size]
        emb_dow = self.dow_embed(dow)        # [B, T, times_embeds_size]
        
        # Concatenate and project
        temporal_concat = torch.cat([emb_user, emb_tod, emb_dow], dim=-1)  # [B, T, temporal_emb_dim]
        temporal_embeds = self.temporal2hidden(temporal_concat)  # [B, T, hidden_dim]
        
        return temporal_embeds
    
    def encode_spatial(self, latlon, place, missing_mask=None):
        """
        Encode spatial features (lat/lon, place)
        
        Args:
            latlon (torch.Tensor): Lat/lon coordinates [B, T, 2]
            place (torch.Tensor): Place IDs [B, T]
            missing_mask (torch.Tensor, optional): Boolean mask for missing observations [B, T]
            
        Returns:
            torch.Tensor: Spatial embeddings [B, T, hidden_dim]
        """
        # Process lat/lon
        sin, cos = torch.sin(latlon), torch.cos(latlon)
        latlon = torch.cat([sin, cos, latlon], dim=-1)
        emb_latlon = self.latlon_proj(latlon)  # [B, T, latlon_emb_dim]
        
        # Get place embeddings
        emb_place = self.place_embed(place)  # [B, T, place_embeds_size]
        
        # Concatenate and project
        spatial_concat = torch.cat([emb_latlon, emb_place], dim=-1)  # [B, T, spatial_emb_dim]
        spatial_embeds = self.spatial2hidden(spatial_concat)  # [B, T, hidden_dim]
        
        # Apply missing observation mask if provided
        if missing_mask is not None:
            # Expand mask to match hidden dimension
            expanded_mask = missing_mask.unsqueeze(-1).expand_as(spatial_embeds)
            # Zero out spatial embeddings for missing observations
            spatial_embeds = spatial_embeds * (~expanded_mask)
        
        return spatial_embeds

    def encode_struct(self, x_enc):
        """
        Encode historical data with separate temporal and spatial components.

        Args:
            x_enc (torch.Tensor): [B, self.token_num * self.token_len, self.token_num] tensor containing 
                                  [user_id, time_of_day, day_of_week, ?, lat, lon, place_id]

        Returns:
            torch.Tensor: [B, self.token_num * self.token_len, hidden_dim] encoded embeddings.
        """
        user_id = x_enc[..., 0].long()
        tod = x_enc[..., 1].long()
        dow = x_enc[..., 2].long()
        latlon = x_enc[..., 4:6].float()
        place = x_enc[..., 6].long()
        
        # Create mask for missing observations (place_id == 40000)
        missing_mask = (place == 40000)
        
        # Encode temporal and spatial components separately
        temporal_embeds = self.encode_temporal(user_id, tod, dow)
        spatial_embeds = self.encode_spatial(latlon, place, missing_mask)
        
        # Combine temporal and spatial embeddings (element-wise addition)
        struct_embeds = temporal_embeds + spatial_embeds
        
        return struct_embeds

    def encode_future_struct(self, x_dec_f):
        """
        Encode future day data (temporal only).

        Args:
            x_dec_f (torch.Tensor): [B, self.token_len, 3] tensor containing 
                                     [user_id, time_of_day, day_of_week]

        Returns:
            torch.Tensor: [B, self.token_len, hidden_dim] encoded embeddings.
        """
        user_id = x_dec_f[..., 0].long()
        tod = x_dec_f[..., 1].long()
        dow = x_dec_f[..., 2].long()
        
        # For future steps, we only have temporal information
        temporal_embeds = self.encode_temporal(user_id, tod, dow)
        
        return temporal_embeds
    

    def aggregate_day(self, enc_embeds):
        """
        Optimized day-level aggregation with parallel processing
        Args:
            enc_embeds (torch.Tensor): [B, self.token_num * self.token_len, hidden_dim]
        Returns:
            torch.Tensor: [B, self.token_num, hidden_dim] aggregated day embeddings
        """
        B, T, H = enc_embeds.size()
        assert T == self.token_num * self.token_len, f"Expected T=self.token_num * self.token_len, got T={T}"
        
        attended_sequences = enc_embeds.view(B * self.token_num, self.token_len, H)
        
        day_embeddings = self.pool_proj(attended_sequences)
        
        # Reshape back to [B, self.token_num, H]
        day_embeds = day_embeddings.view(B, self.token_num, H)
        
        # Final inter-day attention
        day_embeds = self.segment_attention(day_embeds)
        
        return day_embeds
    
    def forward(self, x_enc, x_mark_enc, x_dec_f, x_dec=None, x_mark_dec=None):
        """
        Forward pass of the model.

        Args:
            x_enc (torch.Tensor): [B, self.token_num * self.token_len, self.token_num] historical data.
            x_mark_enc (torch.Tensor): [B, self.token_num, hidden_dim] daily prompt embeddings.
            x_dec_f (torch.Tensor): [B, self.token_len, 3] future day features.
            x_dec (torch.Tensor, optional): Not used. Included for compatibility.
            x_mark_dec (torch.Tensor): [B, hidden_dim] task description prompt.

        Returns:
            torch.Tensor: [B, self.token_len, num_places] logits for place prediction.
        """

        # ------------------------------------------------------------------
        # 1) Encode historical data -> [B, self.token_num * self.token_len, hidden_dim]
        # ------------------------------------------------------------------
        enc_embeds = self.encode_struct(x_enc)  # [B, self.token_num * self.token_len, hidden_dim]
        enc_embeds = self.hierarchical_attention(enc_embeds)
        
        # ------------------------------------------------------------------
        # 2) Aggregate into day-level tokens -> [B, self.token_num, hidden_dim]
        # ------------------------------------------------------------------
        day_embeds = self.aggregate_day(enc_embeds)  # [B, self.token_num, hidden_dim]

        # ------------------------------------------------------------------
        # 3) Add x_mark_enc to day embeddings -> [B, self.token_num, hidden_dim]
        # ------------------------------------------------------------------
        day_embeds = F.normalize(day_embeds, dim=-1)
        x_mark_enc = F.normalize(x_mark_enc, dim=-1)
        
        # day_embeds = day_embeds + x_mark_enc  # [B, self.token_num, hidden_dim]
        day_embeds = day_embeds + self.prompt_fusion1 * x_mark_enc
        
        # ------------------------------------------------------------------
        # 4) Encode future day features -> [B, self.token_len, hidden_dim]
        # ------------------------------------------------------------------
        dec_embeds = self.encode_future_struct(x_dec_f)  # [B, self.token_len, hidden_dim]

        # ------------------------------------------------------------------
        # 5) Add x_mark_dec to each future token -> [B, self.token_len, hidden_dim]
        # ------------------------------------------------------------------

        x_mark_dec_expanded = x_mark_dec.unsqueeze(1).expand(-1, self.token_len, -1)  # [B, self.token_len, hidden_dim]
        
        dec_embeds = F.normalize(dec_embeds, dim=-1)
        x_mark_dec_expanded = F.normalize(x_mark_dec_expanded, dim=-1)
        
        # dec_embeds = dec_embeds + x_mark_dec_expanded  # [B, self.token_len, hidden_dim]
        dec_embeds = dec_embeds + self.prompt_fusion2 * x_mark_dec_expanded

        # ------------------------------------------------------------------
        # 6) Concatenate historical and future embeddings -> [B, 55, hidden_dim]
        # ------------------------------------------------------------------
        full_embeds = torch.cat([day_embeds, dec_embeds], dim=1)  # [B, self.token_num + self.token_len, hidden_dim] = [B, 55, hidden_dim]
        full_embeds = self.full_embed_mlp(full_embeds)  # [B, 55, hidden_dim]

        # ------------------------------------------------------------------
        # 7) Pass through LLaMA
        # ------------------------------------------------------------------
        outputs = self.llama.model(inputs_embeds=full_embeds)
        hidden_states = outputs.last_hidden_state

        # ------------------------------------------------------------------
        # 8) Extract future hidden states -> [B, self.token_len, hidden_dim]
        # ------------------------------------------------------------------
        next_day_states = hidden_states[:, -self.token_len:, :]  # [B, self.token_len, hidden_dim]

        # ------------------------------------------------------------------
        # 9) Apply classification head -> [B, self.token_len, num_places]
        # ------------------------------------------------------------------
        logits = self.classifier(self.dropout(next_day_states))  # [B, self.token_len, num_places]

        return logits
    

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim) 
        )
        
    def forward(self, x):
        x = self.projection(x) 
        return x
    

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=16, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.gate_ffn = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # Multi-head self-attention
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout1(attn_output)
        
        # Feed-forward network
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        gate = torch.sigmoid(self.gate_ffn(x))
        ffn_output = ffn_output * gate
        x = x + self.dropout2(ffn_output)

        return x
    
class MultiLayerAttentionEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=16, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.global_attention = MultiLayerAttentionEncoder(hidden_dim)
        
        self.local_attention = MultiLayerAttentionEncoder(hidden_dim)
        
    def forward(self, x):
        x = self.global_attention(x)
        
        B, T, H = x.shape
        x = x.view(B * self.token_num, self.token_len, H)
        x = self.local_attention(x)
        x = x.view(B, self.token_num * self.token_len, H)
        
        return x
    
class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        
    def forward(self, x):
        return x + self.mlp(self.norm(x))
    
class PoolingModule(nn.Module):
    def __init__(self, hidden_dim, seq_len=token_len, pool_type='linear', dropout=0.1):
        """
        A pooling module that supports multiple pooling strategies.
        
        Args:
            hidden_dim: Hidden dimension size
            seq_len: Sequence length to pool over
            pool_type: Type of pooling: 'attention', 'linear', or 'gru'
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pool_type = pool_type
        
        if pool_type == 'attention':
            # Attention-based pooling
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            
        elif pool_type == 'linear':
            # Linear projection pooling (across time dimension)
            self.linear_pool = nn.Linear(seq_len, 1)
            
        elif pool_type == 'gru':
            # GRU-based pooling
            self.gru = nn.GRU(
                hidden_dim, 
                hidden_dim, 
                num_layers=2, 
                batch_first=True, 
                dropout=dropout, 
                bidirectional=True
            )
            self.gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Pool sequence data into a single vector.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Pooled tensor of shape [batch_size, hidden_dim]
        """
        if self.pool_type == 'attention':
            # Normalize input
            x = self.layer_norm(x)
            
            # Get batch size from input
            batch_size = x.size(0)
            
            # Create query, keys, and values
            query = self.query.expand(batch_size, -1, -1)  # [B, 1, H]
            keys = self.key_proj(x)  # [B, T, H]
            values = self.value_proj(x)  # [B, T, H]
            
            # Compute attention scores
            scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(self.hidden_dim)  # [B, 1, T]
            attn_weights = F.softmax(scores, dim=-1)  # [B, 1, T]
            
            # Apply dropout to attention weights
            attn_weights = self.dropout(attn_weights)
            
            # Weighted sum to get context vector
            context = torch.bmm(attn_weights, values)  # [B, 1, H]
            return context.squeeze(1)  # [B, H]
            
        elif self.pool_type == 'linear':
            # Transpose to [B, H, T] for linear projection across time
            x_t = x.transpose(1, 2)  # [B, H, T]
            pooled = self.linear_pool(x_t).squeeze(-1)  # [B, H]
            return pooled
            
        elif self.pool_type == 'gru':
            # Process sequence with GRU
            _, hidden = self.gru(x)  # hidden: [4, B, H] (2 layers, bidirectional)
            
            # Extract last layer forward and backward hidden states
            forward_hidden = hidden[-2]  # [B, H]
            backward_hidden = hidden[-1]  # [B, H]
            
            # Concatenate and project
            combined = torch.cat([forward_hidden, backward_hidden], dim=-1)  # [B, 2H]
            pooled = self.gru_proj(combined)  # [B, H]
            return pooled

