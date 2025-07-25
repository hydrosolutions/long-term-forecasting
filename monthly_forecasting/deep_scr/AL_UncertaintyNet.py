import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from deep_scr.meta_base import LitMetaForecastBase
import numpy as np

# Shared logging
import logging
from scr.log_config import setup_logging
setup_logging()  

logger = logging.getLogger(__name__) 

def predict_quantile(mu, b, tau, q):
    scale = b / (tau * (1 - tau))
    output = np.where(
        q < tau,
        mu + scale * tau * np.log(2 * q),
        mu - scale * (1 - tau) * np.log(2 * (1 - q))
    )
    return output

class UncertaintyNet(nn.Module):
    def __init__(self,
                 past_dim: int,
                 future_dim: int,
                 static_dim: int,
                 base_learner_dim: int,
                 base_learner_error_dim: int,
                 lookback: int,
                 future_known_steps: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 adaptive_weighting: bool = False,
                 correction_term: bool = False,
                 weight_by_metrics: bool = False,):
        super().__init__()
        self.lookback = lookback
        self.future_known_steps = future_known_steps
        self.base_learner_dim = base_learner_dim
        self.base_learner_error_dim = base_learner_error_dim
        self.out_dim = out_dim
        self.adaptive_weighting = adaptive_weighting
        self.correction_term = correction_term
        self.weight_by_metrics = weight_by_metrics
        if self.weight_by_metrics:
            logger.info("Weighting by metrics is enabled.")

        # per-time-step input dims
        enc_input_dim = past_dim + future_dim + static_dim
        dec_input_dim = future_dim  + static_dim

        # encoder LSTM
        self.encoder = nn.LSTM(
            input_size=enc_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # decoder LSTM
        self.decoder = nn.LSTM(
            input_size=dec_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        base_context_dim = base_learner_dim + base_learner_error_dim
        
        self.base_context_embedding = nn.Sequential(
            nn.Linear(base_context_dim, hidden_dim ),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
        )

        self.combination_net = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Dropout(dropout),
        )

        if self.adaptive_weighting:
            logger.info("Adaptive weighting is enabled.")
            # constructs weights for the base learners [0,1] : Sum = 1
            self.weight_net = nn.Linear(hidden_dim, base_learner_dim)

        if self.correction_term:
            logger.info("Correction term is enabled.")
            # constructs weights for the base learners [0,1] : Sum = 1
            self.correction_net = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        # Prediction head: hidden_dim + 1 -> J outputs
        self.prediction_head = nn.Linear(hidden_dim + 1, 2)
        # Base-learner head: hidden_dim -> K raw scores (shared across outputs)


    def forward(self, batch):
        """
        Args:
        batch with 4 tensors:
            past:          (B, lookback, past_dim)
            future:        (B, lookback+future_known_steps, future_dim)
            static:        (B, static_dim)
            base_learners: (B, base_learner_dim)
        Returns:
            out:           (B, out_dim)
        """
        past = batch['past_input']          # (B, lookback, past_dim)
        future = batch['future_input']      # (B, lookback+future_known_steps, future_dim)
        static = batch['static_input']      # (B, static_dim)
        base_learners = batch['base_learners']  # (B, base_learner_dim)
        base_learner_errors = batch['base_learner_errors']  # (B, base_learner_error_dim)
        
        B = past.size(0)

        # 1) Encoder inputs
        static_enc = static.unsqueeze(1).expand(-1, self.lookback, -1)
        future_enc = future[:, :self.lookback, :]
        enc_input = torch.cat([past, future_enc, static_enc], dim=2)

        _, (h_n, c_n) = self.encoder(enc_input)
        h_enc = h_n[-1].unsqueeze(0)
        c_enc = c_n[-1].unsqueeze(0)

        # 2) Decoder inputs
        static_dec = static.unsqueeze(1).expand(-1, self.future_known_steps, -1)
        future_dec = future[:, self.lookback:, :]
        dec_input = torch.cat([future_dec, static_dec], dim=2)

        _, (h_dec, c_dec) = self.decoder(dec_input, (h_enc, c_enc))
        h_context = h_dec[-1]  # (B, hidden_dim)
        

        # 3) Base-learner inputs
        base_learners = base_learners.flatten(start_dim=1)
        base_learner_errors = base_learner_errors.flatten(start_dim=1)
        base_learners_c = torch.cat([base_learners, base_learner_errors], dim=1)  # (B, lookback+future_known_steps, base_learner_dim)
        base_context = self.base_context_embedding(base_learners_c)  # (B, hidden_dim)
        
        
        h_context =  torch.cat([h_context, base_context], dim=-1)  # (B, 2*hidden_dim)
        h_context = self.combination_net(h_context) # (B, hidden_dim)
        
            
        # 3) Prediction branch
        if self.adaptive_weighting:
            bl_scores = self.weight_net(h_context)
            bl_weights = torch.softmax(bl_scores, dim=1)    # (B, K)
            # after you have bl_weights (B,K) and base_learners (B,K)
            weighted = bl_weights * base_learners        # elementwise (B,K)
            mean_base = weighted.sum(dim=1, keepdim=True)  # (B,1)

        else:
            if self.weight_by_metrics:
                alpha = 1.0
                # Apply softmax to get weights that sum to 1
                bl_weights = torch.softmax(alpha * base_learner_errors, dim=1)
                
                # Apply weights to base learner predictions
                weighted = bl_weights * base_learners  # elementwise (B,K)
                mean_base = weighted.sum(dim=1, keepdim=True)  # (B,1)
            
            else:
                mean_base = torch.mean(base_learners, dim=1).unsqueeze(1)  # (B, 1)
        
        if self.correction_term:
            correction = self.correction_net(h_context)  # (B, 1)
            mean_base = mean_base + correction

        
        h_context = torch.cat([h_context, mean_base], dim=-1)  # (B, hidden_dim+1)
        
        preds = self.prediction_head(h_context)  # (B, 2)

        # split output into b (scale) and tau (asymmetry)
        b_raw = preds[:, 0]   # (B,)
        tau_raw = preds[:, 1]  # (B,)

        # Apply transforms to enforce constraints
        b = F.softplus(b_raw) + 1e-4  # Ensure b > 0 (use softplus to avoid negatives)
        tau = torch.sigmoid(tau_raw)  # Ensure tau in (0, 1) (probability-like)

        # Final output: (mu, b, tau)
        return mean_base.squeeze(1), b, tau  # (B,), (B,), (B,)
    

class AL_Uncertainty_Forecast(LitMetaForecastBase):
    def build_network(self, out_dim: int) -> nn.Module:
        # pull hyperparameters from self.hparams
        return UncertaintyNet(
            past_dim           = self.hparams.past_dim,
            future_dim         = self.hparams.future_dim,
            static_dim         = self.hparams.static_dim,
            base_learner_dim   = self.hparams.base_learner_dim,
            base_learner_error_dim = self.hparams.base_learner_error_dim,
            lookback           = self.hparams.lookback,
            future_known_steps = self.hparams.future_known_steps,
            hidden_dim         = self.hparams.hidden_dim,
            out_dim            = out_dim,
            num_layers         = getattr(self.hparams, 'num_layers', 1),
            dropout            = getattr(self.hparams, 'dropout', 0.0),
            adaptive_weighting  = getattr(self.hparams, 'adaptive_weighting', False),
            correction_term     = getattr(self.hparams, 'correction_term', False),
            weight_by_metrics   = getattr(self.hparams, 'weight_by_metrics', False),
        )
    
    def forward(self, batch):
        # net should output shape (B, out_dim)
        mu, b, tau = self.net(batch)
        return mu, b, tau  # (B,), (B,), (B,)
    
    def training_step(self, batch, batch_idx):
        mu, b, tau = self(batch)
        y_true = batch['y']

        loss = self.criterion(mu, b, tau, y_true)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # net should output shape (B, out_dim)
        mu, b, tau = self(batch)
        # compute loss
        y_true = batch['y']
        loss = self.criterion(mu, b, tau, y_true)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # net should output shape (B, out_dim)
        mu, b, tau = self(batch)
        # compute loss
        y_true = batch['y']
        loss = self.criterion(mu, b, tau, y_true)
        self.log('test_loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        # 1) forward
        mu, b, tau = self(batch)  # Outputs (B,), (B,), (B,)

        mu = mu.detach().cpu()
        b = b.detach().cpu()
        tau = tau.detach().cpu()

        # 2) extract date components & code
        day_np   = batch['day'].detach().cpu().numpy().ravel().astype(int)
        month_np = batch['month'].detach().cpu().numpy().ravel().astype(int)
        year_np  = batch['year'].detach().cpu().numpy().ravel().astype(int)

        if isinstance(batch['code'], torch.Tensor):
            code_np = batch['code'].detach().cpu().numpy().ravel()
        else:
            code_np = list(batch['code'])

        # 3) build pandas dates
        dates = pd.to_datetime({'year': year_np,
                                'month': month_np,
                                'day': day_np})

        # 4) assemble DataFrame
        data = {'date': dates, 'code': code_np}


        # 5) Predict quantiles using predict_quantile
        data['Q_mean'] = mu.numpy()
        if self.quantiles is not None:
            for q in self.quantiles:
                q_pred = predict_quantile(mu, b, tau, q)  # (B,)
                data[f"Q{int(q*100)}"] = q_pred
        else:
            # If no quantiles given, output just the median
            data['Q50'] = mu.numpy()

        return pd.DataFrame(data)