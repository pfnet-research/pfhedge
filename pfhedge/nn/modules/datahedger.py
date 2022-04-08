from typing import List
from tqdm import tqdm  
import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader

from pfhedge.nn import EntropicRiskMeasure
from pfhedge.nn import HedgeLoss

class MarketDataset(Dataset):
    """Market information dataset.
    Args:
        - data_set (List[Tensor]): [prices, information, payoff]
    Shape:
        - prices: (N, T+1, n_asset)
        - information: (N, T or T+1, n_feature)
        - payoff: (N, 1)
    """
    def __init__(self, data_set):
        # TODO(songyan): more general payoff (path dependent)
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
        
    """ Hedger to hedge with only data generated but not the generating class
    Args:
        - model (torch.nn.Module) or models (List[Module]): depending on independent neural network at each time step or the same neural network at each time
        - dataset_market (Dataset)
        - criterion (HedgeLoss)
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
        # TODO(songyan): 1. cost configuration  2. state_information configuration 3. 
        """Compute the terminal wealth

        Args:
            input = prices, information, payoff
        
        Note:
            V_t: Wealth process
            I_t: Information process = (information, state_information)
            H: hedging strategy
            S_t: Price process
            C_t: Cost process 

            dV_t = H(I_t)dS_t - dC_t 

        Returns:
            V_T: torch.Tensor
        """

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
        # TODO(songyan): path dependent payoff
        
        """Compute the PnL

        Args:
            input = prices, information, payoff
        
        Note:
            PnL = V_T - h(S_T)

        Returns:
            PnL: torch.Tensor
        """
        prices, information, payoff = input
        wealth = self(input)
        pnl = wealth - payoff
        return pnl

    def compute_loss(self,input):
        """Compute the loss

        Args:
            input = prices, information, payoff
        
        Note:
            loss = loss_functional(PnL)

        Returns:
            loss: torch.Tensor
        """
        loss = self.criterion(self.compute_pnl(input))
        return loss

        
    def fit(self, EPOCHS = 1, batch_size = 256, optimizer = torch.optim.Adam, lr = 0.005):
        """Fitting process

        Args:
            EPOCHS
            batch_size = 256
            optimizer = torch.optim.Adam
        
        Note:
            loss = loss_functional(PnL)

        Returns:
            loss: torch.Tensor
        """
        self.dataloader_market= DataLoader(self.dataset_market, batch_size=batch_size, shuffle=True, num_workers=0)

        self.optimizer = optimizer(self.parameters(), lr=lr)
        history = []
        progress = tqdm(range(EPOCHS))
        for _ in progress:
            self.train(True)
            for i, data in enumerate(self.dataloader_market):
                self.optimizer.zero_grad()
                loss = self.compute_loss(data)
                loss.backward()
                self.optimizer.step()
                history.append(loss.item())
                progress.desc = "Loss=" + str(loss.item())

        return history
            
    
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
        """Compute the terminal wealth

        Args:
            input = prices, information, payoff
        
        Note:
            V_t: Wealth process
            I_t: Information process = (information, state_information)
            H: hedging strategy
            S_t: Price process
            C_t: Cost process 

            dV_t = H_t(I_t)dS_t - dC_t 

        Returns:
            V_T: torch.Tensor
        """
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
 




                        