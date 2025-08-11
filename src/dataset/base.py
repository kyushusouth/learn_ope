from abc import ABCMeta, abstractmethod


class BaseBanditDataset(metaclass=ABCMeta):
    @abstractmethod
    def obtain_batch_bandit_feedback(self) -> None:
        raise NotImplementedError
