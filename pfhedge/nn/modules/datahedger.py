from typing import List
from tqdm import tqdm  
import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader

from pfhedge.nn import EntropicRiskMeasure
from pfhedge.nn import HedgeLoss

class MarketDataset(Dataset):
    """Price dataset."""
    def __init__(self, data_set):
        # TODO(songyan): more general payoff (path dependent)
        """
        Args:
            data_set (List[Tensor]): [prices, information, payoff]
        Shape:
            - prices: (N, T+1, n_asset)
            - information: (N, T or T+1, n_feature)
            - payoff: (N, 1)
        """
        self.prices, self.information, self.payoff = data_set
        self.data = [self.prices, self.information, self.payoff]
        self.n_asset = int(self.prices.shape[-1])
        self.n_feature = int(self.information.shape[-1])
        self.T = int(self.prices.shape[1]) - 1

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx):
        return [self.prices[idx], self.information[idx], self.payoff[idx]]


class DataHedger(Module):
        
    """
        Args:
        model (torch.nn.Module)
        dataset_market (Dataset)
        criterion (HedgeLoss)

    """

    def __init__(
        self, model: Module, 
        dataset_market: MarketDataset,
        criterion: HedgeLoss = EntropicRiskMeasure()
    ):
        super().__init__()
        self.model = model
        self.dataset_market = dataset_market
        self.criterion = criterion
        
    def forward(self, input) -> Tensor:
        prices, information, payoff = input
        T = int(prices.shape[1]) - 1
        wealth = 0
        holding = torch.zeros_like(prices[:,0,:])
        for t in range(T):
            state_information = holding
            all_information = torch.cat([information[:,t,:],state_information], axis = -1)
            if self.model:
                holding = self.model(all_information)
            else:
                holding = self.models[t](all_information)
            wealth = wealth + holding * (prices[:,t+1,:] - prices[:,t,:])
        cost = 0
        wealth= torch.sum(wealth, axis = -1, keepdim=True) - cost
        return wealth
        
    def compute_pnl(self, input) -> Tensor:
        prices, information, payoff = input
        wealth = self(input)
        pnl = wealth - payoff
        return pnl

    def compute_loss(self,input):
        loss = self.criterion(self.compute_pnl(input))
        return loss

    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(self.dataloader_market):
            self.optimizer.zero_grad()
            loss = self.compute_loss(data)
            loss.backward()
            self.optimizer.step()
        return loss.item()
        
    def fit(self, EPOCHS):

        self.dataloader_market= DataLoader(self.dataset_market, batch_size=10000,
                        shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam
        self.optimizer = optimizer(self.parameters(), lr=0.0001)

        progress = tqdm(range(EPOCHS))
        for _ in progress:
            self.train(True)
            avg_loss = self.train_one_epoch(1)
            progress.desc = "Loss=" + str(avg_loss)
            
    
    def pricer(self) -> Tensor:
        if not hasattr(self, 'pnl'):
            self.pnl = self.compute_pnl(self.dataset_market.data)
        self.price = -self.criterion.cash(self.pnl)
        return self.price




class DeepHedger(DataHedger):
        
    """
        Args:
        model (torch.nn.ModuleList)
        dataset_market (Dataset)
        criterion (HedgeLoss)

    """

    def __init__(
        self, models: List, 
        dataset_market: MarketDataset,
        criterion: HedgeLoss = EntropicRiskMeasure()
    ):
        super().__init__(None, dataset_market, criterion)
        self.models = ModuleList(models)
        
    def forward(self, input) -> Tensor:
        prices, information, payoff = input
        T = int(prices.shape[1]) - 1
        wealth = 0
        holding = torch.zeros_like(prices[:,0,:])
        if not len(self.models) == T:
            raise ValueError("models must has the same length of time steps")
        for t in range(T):
            state_information = holding
            all_information = torch.cat([information[:,t,:],state_information], axis = -1)
            holding = self.models[t](all_information)
            wealth = wealth + holding * (prices[:,t+1,:] - prices[:,t,:])
        cost = 0
        wealth= torch.sum(wealth, axis = -1, keepdim=True) - cost
        return wealth
 
                        